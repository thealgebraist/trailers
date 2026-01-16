const std = @import("std");
const http = std.http;
const json = std.json;
const fs = std.fs;

const PypiMetadata = struct {
    info: struct {
        name: []const u8,
        version: []const u8,
        requires_dist: ?[][]const u8 = null,
    },
    urls: []struct {
        url: []const u8,
        packagetype: []const u8,
        filename: []const u8,
    },
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 3) {
        std.debug.print("Usage: zpip install <package> --dest <dir>\n", .{});
        return;
    }

    var pkg_name: []const u8 = "";
    var dest_dir: []const u8 = "site-packages";

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "install")) {
            i += 1;
            if (i < args.len) pkg_name = args[i];
        } else if (std.mem.eql(u8, args[i], "--dest")) {
            i += 1;
            if (i < args.len) dest_dir = args[i];
        }
    }

    if (pkg_name.len == 0) {
        std.debug.print("Error: No package specified.\n", .{});
        return;
    }

    try fs.cwd().makePath(dest_dir);
    
    var visited = std.StringHashMap(void).init(allocator);
    defer visited.deinit();

    std.debug.print("--- zpip: Installing {s} to {s} ---\\n", .{ pkg_name, dest_dir });
    try installRecursive(allocator, pkg_name, dest_dir, &visited);
    std.debug.print("\nSuccessfully installed all dependencies.\n", .{});
}

fn installRecursive(allocator: std.mem.Allocator, pkg: []const u8, dest: []const u8, visited: *std.StringHashMap(void)) !void {
    // Basic normalization: lowercase
    const normalized_name = try allocator.dupe(u8, pkg);
    defer allocator.free(normalized_name);
    for (normalized_name) |*c| c.* = std.ascii.toLower(c.*);

    if (visited.contains(normalized_name)) return;
    try visited.put(try allocator.dupe(u8, normalized_name), {});

    std.debug.print("Resolving: {s}...\\n", .{pkg});

    // 1. Fetch Metadata
    const url = try std.fmt.allocPrint(allocator, "https://pypi.org/pypi/{s}/json", .{pkg});
    defer allocator.free(url);

    var client = http.Client{ .allocator = allocator };
    defer client.deinit();

    var buf: [1024 * 512]u8 = undefined; // 512KB for JSON
    var req = try client.open(.GET, try std.Uri.parse(url), .{ .server_header_buffer = &buf });
    defer req.deinit();
    try req.send();
    try req.wait();

    const body = try req.reader().readAllAlloc(allocator, 1024 * 1024 * 2); // 2MB max
    defer allocator.free(body);

    const parsed = try json.parseFromSlice(PypiMetadata, allocator, body, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();

    // 2. Find Best Wheel
    var download_url: ?[]const u8 = null;
    var filename: ?[]const u8 = null;
    for (parsed.value.urls) |u| {
        if (std.mem.eql(u8, u.packagetype, "bdist_wheel")) {
            download_url = u.url;
            filename = u.filename;
            break;
        }
    }

    if (download_url) |d_url| {
        std.debug.print("Downloading {s}...\\n", .{filename.?});
        const full_dest_path = try fs.path.join(allocator, &.{ dest, filename.? });
        defer allocator.free(full_dest_path);

        // Download using curl for speed and simplicity in this PoC
        const child_args = [_][]const u8{ "curl", "-L", "-o", full_dest_path, d_url };
        var child = std.ChildProcess.init(&child_args, allocator);
        _ = try child.spawnAndWait();

        // PoC: We just place the wheels in --dest. 
        // In a full implementation, we'd unzip them here.
    }

    // 3. Recurse Dependencies
    if (parsed.value.info.requires_dist) |deps| {
        for (deps) |dep_str| {
            // Very simple parser for "name (>=version) ; extra"
            var it = std.mem.splitAny(u8, dep_str, " (;<=>");
            const dep_name = it.first();
            if (dep_name.len > 0) {
                try installRecursive(allocator, dep_name, dest, visited);
            }
        }
    }
}

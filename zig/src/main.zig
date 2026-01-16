const std = @import("std");
const c = @cImport({
    @cInclude("onnxruntime_c_api.h");
});

// --- WAV Utility ---
const WavHeader = struct {
    riff: [4]u8 = "RIFF".*,
    file_size: u32,
    wave: [4]u8 = "WAVE".*,
    fmt: [4]u8 = "fmt ".*,
    fmt_size: u32 = 16,
    format: u16 = 1,
    channels: u16,
    sample_rate: u32,
    byte_rate: u32,
    block_align: u16,
    bps: u16 = 16,
    data: [4]u8 = "data".*,
    data_size: u32,
};

fn writeWav(path: []const u8, samples: []const f32, sample_rate: u32) !void {
    const file = try std.fs.cwd().createFile(path, .{});
    defer file.close();

    const data_size = @as(u32, @intCast(samples.len * 2));
    const header = WavHeader{
        .file_size = 36 + data_size,
        .channels = 1,
        .sample_rate = sample_rate,
        .byte_rate = sample_rate * 2,
        .block_align = 2,
        .data_size = data_size,
    };

    try file.writeAll(std.mem.asBytes(&header));
    for (samples) |s| {
        const pcm = @as(i16, @intFromFloat(@max(-1.0, @min(1.0, s)) * 32767.0));
        try file.writeAll(std.mem.asBytes(&pcm));
    }
}

// --- Generator Logic ---
const SloppyGenerator = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) SloppyGenerator {
        return .{ .allocator = allocator };
    }

    pub fn generateSfx(self: *SloppyGenerator, path: []const u8) !void {
        std.debug.print("Zig: Generating Glitch SFX -> {s}\n", .{path});
        const count = 22050 * 2; // 2 seconds
        var samples = try self.allocator.alloc(f32, count);
        defer self.allocator.free(samples);

        var prng = std.Random.DefaultPrng.init(@intCast(std.time.milliTimestamp()));
        const random = prng.random();

        for (0..count) |i| {
            const t = @as(f32, @floatFromInt(i)) / 22050.0;
            // FM synthesis + Bit crushing logic
            const carrier = @sin(t * 200.0 * 2.0 * std.math.pi);
            const mod = @sin(t * 50.0 * 2.0 * std.math.pi) * 10.0;
            var s = @sin((t * 100.0 + mod) * 2.0 * std.math.pi) * carrier;
            
            // Add some "sloppy" digital noise
            if (i % 10 == 0) s += random.float(f32) * 0.2;
            samples[i] = s;
        }
        try writeWav(path, samples, 22050);
    }

    pub fn generateMusic(self: *SloppyGenerator, path: []const u8) !void {
        std.debug.print("Zig: Generating Dark Ambient -> {s}\n", .{path});
        const count = 22050 * 5; // 5 seconds
        var samples = try self.allocator.alloc(f32, count);
        defer self.allocator.free(samples);

        for (0..count) |i| {
            const t = @as(f32, @floatFromInt(i)) / 22050.0;
            const drone = @sin(t * 40.0 * 2.0 * std.math.pi) * 0.4 + @sin(t * 41.0 * 2.0 * std.math.pi) * 0.2;
            const glitch = if (@as(u32, @intCast(i)) % 4410 < 100) @sin(t * 1000.0) else 0.0;
            samples[i] = (drone + glitch) * 0.8;
        }
        try writeWav(path, samples, 22050);
    }

    pub fn generateVo(self: *SloppyGenerator, text: []const u8, path: []const u8) !void {
        _ = text;
        std.debug.print("Zig: Voice Over (ONNX Harness) -> {s}\n", .{path});
        // Here we would call the ONNX Runtime C API to load Piper or Bark.
        // For the minimal Zig release, we produce a unique glitch voice.
        try self.generateSfx(path); 
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 3) {
        std.debug.print("Usage: zig-sloppy <sfx|music|vo> <output_path> [text]\n", .{});
        return;
    }

    const mode = args[1];
    const path = args[2];
    var gen = SloppyGenerator.init(allocator);

    if (std.mem.eql(u8, mode, "sfx")) {
        try gen.generateSfx(path);
    } else if (std.mem.eql(u8, mode, "music")) {
        try gen.generateMusic(path);
    } else if (std.mem.eql(u8, mode, "vo")) {
        const text = if (args.len > 3) args[3] else "Hello";
        try gen.generateVo(text, path);
    } else {
        std.debug.print("Unknown mode: {s}\n", .{mode});
    }
}

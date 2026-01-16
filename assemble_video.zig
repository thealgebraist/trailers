const std = @import("std");
const process = std.process;
const fs = std.fs;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try process.argsAlloc(allocator);
    defer process.argsFree(allocator, args);

    if (args.len < 3) {
        std.debug.print("Usage: {s} <assets_dir> <output_file>\n", .{args[0]});
        return;
    }

    const assets_dir = args[1];
    const output_file = args[2];

    const scenes = [_][]const u8{
        "01_skaro_landscape", "02_dalek_army", "03_dalek_close_eye", "04_rusty_lonely",
        "05_memory_glitch", "06_cornfield_reveal", "07_farmhouse_exterior", "08_baby_dalek_basket",
        "09_nana_pop_holding", "10_baking_pie", "11_fishing_trip", "12_tractor_ride",
        "13_school_exterior", "14_classroom_learning", "15_school_friends", "16_class_photo",
        "17_reading_book", "18_fishing_success", "19_prom_night", "20_first_crush",
        "21_sky_darkens", "22_rusty_looks_up", "23_leaving_home", "24_back_on_skaro",
        "25_supreme_dalek", "26_pie_reveal", "27_dalek_confusion", "28_pie_in_face",
        "29_escape", "30_reunion", "31_title_card", "32_post_credits",
    };

    var ffmpeg_args = std.ArrayList([]const u8).init(allocator);
    defer ffmpeg_args.deinit();

    try ffmpeg_args.append("ffmpeg");
    try ffmpeg_args.append("-y");

    // 1. Add Image Inputs (Inputs 0-31)
    for (scenes) |id| {
        const img_path = try std.fmt.allocPrint(allocator, "{s}/images/{s}.png", .{ assets_dir, id });
        try ffmpeg_args.append("-loop"); try ffmpeg_args.append("1");
        try ffmpeg_args.append("-t"); try ffmpeg_args.append("4");
        try ffmpeg_args.append("-i"); try ffmpeg_args.append(img_path);
    }

    // 2. Add VO Inputs (Inputs 32-63)
    for (scenes) |id| {
        const vo_path = try std.fmt.allocPrint(allocator, "{s}/voice/{s}.wav", .{ assets_dir, id });
        var file_exists = true;
        std.fs.cwd().access(vo_path, .{}) catch { file_exists = false; };

        if (file_exists) {
            try ffmpeg_args.append("-i"); try ffmpeg_args.append(vo_path);
        } else {
            // Use a 4-second silent audio source if VO is missing
            try ffmpeg_args.append("-f"); try ffmpeg_args.append("lavfi");
            try ffmpeg_args.append("-t"); try ffmpeg_args.append("4");
            try ffmpeg_args.append("-i"); try ffmpeg_args.append("anullsrc=r=44100:cl=stereo");
        }
    }

    // 3. Add Music Inputs (Inputs 64-66)
    const music_themes = [_][]const u8{ "theme_dark", "theme_acoustic", "theme_epic" };
    for (music_themes) |id| {
        const m_path = try std.fmt.allocPrint(allocator, "{s}/music/{s}.wav", .{ assets_dir, id });
        try ffmpeg_args.append("-i"); try ffmpeg_args.append(m_path);
    }

    var filter = std.ArrayList(u8).init(allocator);
    defer filter.deinit();

    // Visual Filter: Scaling and Concat
    for (0..32) |i| {
        try filter.writer().print("[{d}:v]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1[v{d}];", .{ i, i });
    }
    for (0..32) |i| { try filter.writer().print("[v{d}]", .{i}); }
    try filter.writer().print("concat=n=32:v=1:a=0[vout];", .{});

    // Audio Filter: Resample and Concat VOs
    for (32..64) |i| {
        try filter.writer().print("[{d}:a]aresample=44100,aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[a{d}];", .{ i, i });
    }
    for (32..64) |i| { try filter.writer().print("[a{d}]", .{i}); }
    try filter.writer().print("concat=n=32:v=0:a=1[vo_full];", .{});

    // Music Layering
    // We assume music inputs 64, 65, 66 exist and are looped/trimmed to fit
    try filter.writer().print("[64:a][65:a][66:a]amix=inputs=3:duration=first[music_mix];", .{});
    
    // Final Mix: VO (loud) + Music (quiet)
    try filter.writer().print("[vo_full][music_mix]amix=inputs=2:weights=1 0.2:duration=first[aout]", .{});

    try ffmpeg_args.append("-filter_complex");
    try ffmpeg_args.append(filter.items);

    try ffmpeg_args.append("-map"); try ffmpeg_args.append("[vout]");
    try ffmpeg_args.append("-map"); try ffmpeg_args.append("[aout]");

    // MPEG2 95% Quality Settings
    try ffmpeg_args.append("-c:v"); try ffmpeg_args.append("mpeg2video");
    try ffmpeg_args.append("-q:v"); try ffmpeg_args.append("2");
    try ffmpeg_args.append("-c:a"); try ffmpeg_args.append("mp2");
    try ffmpeg_args.append("-b:a"); try ffmpeg_args.append("384k");
    try ffmpeg_args.append("-ar"); try ffmpeg_args.append("48000");
    try ffmpeg_args.append("-ac"); try ffmpeg_args.append("2");
    
    try ffmpeg_args.append(output_file);

    std.debug.print("Executing Full Production Audio Mix...\n", .{});
    var child = process.Child.init(ffmpeg_args.items, allocator);
    _ = try child.spawnAndWait();
}
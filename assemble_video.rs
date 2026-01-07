use std::process::Command;
include!("assemble_video_config.rs");

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: ./assemble_video_rust <assets_dir> <output_file> <project_type>");
        std::process::exit(1);
    }

    let assets_dir = &args[1];
    let output_file = &args[2];
    let project_type = &args[3];

    let config = get_config(project_type);
    let fps = 24;
    let width = 1280;
    let height = 720;
    let total_duration = 120.0;

    println!("--- Assembling {} trailer (Rust/Audio Fix) ---", config.name);

    let mut cmd = Command::new("ffmpeg");
    cmd.arg("-y");

    let mut input_count: usize = 0;

    // 1. Add Image Inputs (Indices 0 to n-1)
    for s_id in &config.scenes {
        let path = format!("{}/images/{}.png", assets_dir, s_id);
        if std::path::Path::new(&path).exists() {
            println!("  Found Image: {}", path);
            cmd.arg("-framerate").arg(fps.to_string()).arg("-loop").arg("1").arg("-t").arg(config.scene_duration.to_string()).arg("-i").arg(path);
        } else {
            println!("  MISSING Image: {} (Using black fallback)", path);
            cmd.arg("-f").arg("lavfi").arg("-t").arg(config.scene_duration.to_string()).arg("-i").arg(format!("color=c=black:s={}x{}", width, height));
        }
        input_count += 1;
    }
    let n_scenes = config.scenes.len();

    // 2. Add Voiceover Input (Index n)
    let vo_path = format!("{}/voice/voiceover_full.wav", assets_dir);
    if std::path::Path::new(&vo_path).exists() {
        println!("  Found Voiceover: {}", vo_path);
        cmd.arg("-i").arg(&vo_path);
    } else {
        println!("  MISSING Voiceover: {} (Using silence)", vo_path);
        cmd.arg("-f").arg("lavfi").arg("-t").arg(total_duration.to_string()).arg("-i").arg("anullsrc=r=44100:cl=stereo");
    }
    let vo_idx = input_count;
    input_count += 1;

    // 3. Collect SFX Inputs
    let mut available_sfx = Vec::new();
    for s_id in &config.scenes {
        let path = format!("{}/sfx/{}.wav", assets_dir, s_id);
        if std::path::Path::new(&path).exists() {
            println!("  Found SFX: {}", path);
            cmd.arg("-i").arg(path);
            available_sfx.push(input_count);
            input_count += 1;
        }
    }

    // 4. Collect Music Inputs
    let mut available_music = Vec::new();
    for theme in &config.music_themes {
        let path = format!("{}/music/{}.wav", assets_dir, theme);
        if std::path::Path::new(&path).exists() {
            println!("  Found Music: {}", path);
            cmd.arg("-i").arg(path);
            available_music.push(input_count);
            input_count += 1;
        } else {
            println!("  MISSING Music: {}", path);
        }
    }

    // 5. Build Filter Complex
    let mut filter = String::new();

    // Visual: Static Montage
    for i in 0..n_scenes {
        filter.push_str(&format!(
            "[{}:v]scale=w={}:h={}:force_original_aspect_ratio=decrease,pad={}:{}:(ow-iw)/2:(oh-ih)/2,setsar=1[v{}];",
            i, width, height, width, height, i
        ));
    }
    for i in 0..n_scenes { filter.push_str(&format!("[v{}]", i)); }
    filter.push_str(&format!("concat=n={}:v=1:a=0[vout];", n_scenes));

    // Audio Logic
    let mut mix_inputs = Vec::new();

    // Layer 1: VO
    filter.push_str(&format!("[{}:a]aresample=44100,aformat=sample_fmts=s16:sample_rates=44100:channel_layouts=stereo,volume=1.2[vov];", vo_idx));
    mix_inputs.push("[vov]");

    // Layer 2: SFX
    if !available_sfx.is_empty() {
        for (i, &idx) in available_sfx.iter().enumerate() {
            filter.push_str(&format!("[{}:a]aresample=44100,aformat=sample_fmts=s16:sample_rates=44100:channel_layouts=stereo[sfx_clip{}];", idx, i));
        }
        for i in 0..available_sfx.len() { filter.push_str(&format!("[sfx_clip{}]", i)); }
        filter.push_str(&format!("amix=inputs={}:duration=longest,volume=1.0[sfx_layer];", available_sfx.len()));
        mix_inputs.push("[sfx_layer]");
    }

    // Layer 3: Music
    if !available_music.is_empty() {
        for (i, &idx) in available_music.iter().enumerate() {
            filter.push_str(&format!("[{}:a]aloop=loop=-1:size=2e+09,aresample=44100,aformat=sample_fmts=s16:sample_rates=44100:channel_layouts=stereo[m_clip{}];", idx, i));
        }
        for i in 0..available_music.len() { filter.push_str(&format!("[m_clip{}]", i)); }
        filter.push_str(&format!("amix=inputs={}:duration=longest:normalize=0,volume=0.5[music_layer];", available_music.len()));
        mix_inputs.push("[music_layer]");
    }

    // Final Audio Mix
    let mix_str = mix_inputs.join("");
    // FIXED: Corrected the format string placeholder and removed invalid space
    filter.push_str(&format!("{}amix=inputs={}:duration=longest:normalize=0[aout]", mix_str, mix_inputs.len()));

    cmd.arg("-filter_complex").arg(filter);
    cmd.arg("-map").arg("[vout]").arg("-map").arg("[aout]");
    cmd.arg("-t").arg(total_duration.to_string());
    
    cmd.arg("-c:v").arg("mpeg2video").arg("-q:v").arg("2").arg("-b:v").arg("15000k");
    cmd.arg("-c:a").arg("mp2").arg("-b:a").arg("384k");
    cmd.arg("-pix_fmt").arg("yuv420p").arg("-r").arg(fps.to_string()).arg("-f").arg("mpeg");
    cmd.arg(output_file);

    let status = cmd.status().expect("Failed to execute FFmpeg");
    if status.success() {
        println!("Successfully assembled: {}", output_file);
    } else {
        eprintln!("FFmpeg failed with status: {}", status);
    }
}
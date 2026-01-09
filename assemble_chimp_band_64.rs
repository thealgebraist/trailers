use std::process::Command;
include!("assemble_video_config.rs");

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: ./assemble_chimp_band_64 <assets_dir> <output_file>");
        std::process::exit(1);
    }

    let assets_dir = &args[1];
    let output_file = &args[2];
    let project_type = "chimp_band_64";

    let config = get_config(project_type);
    let fps = 24;
    let width = 1280;
    let height = 720;
    
    // Calculate total duration based on scenes
    let total_duration = config.scenes.len() as f32 * config.scene_duration;

    println!("--- Assembling {} (Rust/FFmpeg - No Music) ---", config.name);

    let mut cmd = Command::new("ffmpeg");
    cmd.arg("-y");

    let mut input_count: usize = 0;

    // 1. Add Image Inputs
    for s_id in &config.scenes {
        let path = format!("{}/images/{}.png", assets_dir, s_id);
        if std::path::Path::new(&path).exists() {
            cmd.arg("-framerate").arg(fps.to_string())
               .arg("-loop").arg("1")
               .arg("-t").arg(config.scene_duration.to_string())
               .arg("-i").arg(path);
        } else {
            cmd.arg("-f").arg("lavfi")
               .arg("-t").arg(config.scene_duration.to_string())
               .arg("-i").arg(format!("color=c=black:s={}x{}", width, height));
        }
        input_count += 1;
    }
    let n_scenes = config.scenes.len();

    // 2. Add SFX Inputs (one for each scene)
    let sfx_start_idx = input_count;
    for s_id in &config.scenes {
        let path = format!("{}/sfx/{}.wav", assets_dir, s_id);
        if std::path::Path::new(&path).exists() {
            cmd.arg("-i").arg(path);
        } else {
            cmd.arg("-f").arg("lavfi")
               .arg("-t").arg(config.scene_duration.to_string())
               .arg("-i").arg("anullsrc=r=44100:cl=stereo");
        }
        input_count += 1;
    }

    // 3. Build Filter Complex
    let mut filter = String::new();

    // Visual: Concat scenes with fixed aspect ratio and no zoom
    for i in 0..n_scenes {
        filter.push_str(&format!(
            "[{}:v]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1[v{}];",
            i, i
        ));
    }
    for i in 0..n_scenes { filter.push_str(&format!("[v{}]", i)); }
    filter.push_str(&format!("concat=n={}:v=1:a=0[vout];", n_scenes));

    // SFX: Concat SFX to match scenes
    for i in 0..n_scenes {
        let idx = sfx_start_idx + i;
        filter.push_str(&format!("[{}:a]aresample=44100,aformat=sample_fmts=s16:sample_rates=44100:channel_layouts=stereo,atrim=0:{}[sfx{}];", 
            idx, config.scene_duration, i));
    }
    for i in 0..n_scenes { filter.push_str(&format!("[sfx{}]", i)); }
    filter.push_str(&format!("concat=n={}:v=0:a=1[aout];", n_scenes));

    cmd.arg("-filter_complex").arg(filter);
    cmd.arg("-map").arg("[vout]").arg("-map").arg("[aout]");
    cmd.arg("-t").arg(total_duration.to_string());
    
    if output_file.ends_with(".mp4") {
        cmd.arg("-c:v").arg("libx264").arg("-preset").arg("fast").arg("-crf").arg("18");
        cmd.arg("-c:a").arg("aac").arg("-b:a").arg("192k");
        cmd.arg("-pix_fmt").arg("yuv420p");
    } else {
        cmd.arg("-c:v").arg("mpeg2video").arg("-q:v").arg("2").arg("-b:v").arg("15000k");
        cmd.arg("-c:a").arg("mp2").arg("-b:a").arg("384k");
        cmd.arg("-pix_fmt").arg("yuv420p").arg("-r").arg(fps.to_string()).arg("-f").arg("mpeg");
    }
    cmd.arg(output_file);

    println!("Executing FFmpeg (no music)...");
    let status = cmd.status().expect("Failed to execute FFmpeg");
    if status.success() {
        println!("Successfully assembled: {}", output_file);
    } else {
        eprintln!("FFmpeg failed with status: {}", status);
    }
}
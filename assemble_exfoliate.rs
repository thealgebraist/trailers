use std::process::Command;
include!("assemble_video_config.rs");

fn get_duration(path: &str) -> f32 {
    let output = Command::new("ffprobe")
        .args(&["-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path])
        .output().ok();
    if let Some(out) = output {
        if let Ok(s) = std::str::from_utf8(&out.stdout) {
            return s.trim().parse::<f32>().unwrap_or(3.0);
        }
    }
    3.0
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: ./assemble_exfoliate <assets_dir> <output_file>");
        std::process::exit(1);
    }

    let assets_dir = &args[1];
    let output_file = &args[2];
    
    let fps = 24;
    let width = 1280;
    let height = 720;

    println!("--- Assembling EXFOLIATE (Sync Logic - No Music) ---");

    // 1. Probe all voice durations
    let mut voice_durations = Vec::new();
    for i in 0..66 {
        let path = format!("{}/voice/voice_{:02}.wav", assets_dir, i);
        if std::path::Path::new(&path).exists() {
            voice_durations.push(get_duration(&path));
        } else {
            voice_durations.push(3.0);
        }
    }

    let mut cmd = Command::new("ffmpeg");
    cmd.arg("-y");

    // 2. Add Video Inputs (130 total)
    let before_dur: f32 = 1.5;
    // Intro 00, 01
    for i in 0..2 {
        let path = format!("{}/images/{:02}_scene.png", assets_dir, i);
        let mut dur = voice_durations[i];
        let target_dur = if i == 0 { 7.0 } else { 8.0 };
        if dur < target_dur { dur = target_dur; }
        
        if std::path::Path::new(&path).exists() {
            cmd.arg("-loop").arg("1").arg("-t").arg(dur.to_string()).arg("-i").arg(path);
        } else {
            cmd.arg("-f").arg("lavfi").arg("-t").arg(dur.to_string()).arg("-i").arg(format!("color=c=black:s={}x{}", width, height));
        }
        // Update voice_durations so audio matches
        voice_durations[i] = dur;
    }
    // Positions 02-65
    for i in 2..66 {
        let dur = voice_durations[i];
        let d_half = dur / 2.0;
        
        let before_path = format!("{}/images/{:02}_before.png", assets_dir, i);
        let during_path = format!("{}/images/{:02}_during.png", assets_dir, i);

        // Before
        if std::path::Path::new(&before_path).exists() {
            cmd.arg("-loop").arg("1").arg("-t").arg(d_half.to_string()).arg("-i").arg(before_path);
        } else {
            cmd.arg("-f").arg("lavfi").arg("-t").arg(d_half.to_string()).arg("-i").arg(format!("color=c=black:s={}x{}", width, height));
        }
        // During
        if std::path::Path::new(&during_path).exists() {
            cmd.arg("-loop").arg("1").arg("-t").arg((dur - d_half).to_string()).arg("-i").arg(during_path);
        } else {
            cmd.arg("-f").arg("lavfi").arg("-t").arg((dur - d_half).to_string()).arg("-i").arg(format!("color=c=black:s={}x{}", width, height));
        }
    }

    // 3. Add Voice Inputs (66 total)
    for i in 0..66 {
        let path = format!("{}/voice/voice_{:02}.wav", assets_dir, i);
        if std::path::Path::new(&path).exists() {
            cmd.arg("-i").arg(path);
        } else {
            let dur = voice_durations[i];
            cmd.arg("-f").arg("lavfi").arg("-t").arg(dur.to_string()).arg("-i").arg("anullsrc=r=44100:cl=stereo");
        }
    }

    // 4. Add SFX Inputs (66 total)
    for i in 0..66 {
        let path = format!("{}/sfx/{:02}_exfoliate.wav", assets_dir, i);
        if std::path::Path::new(&path).exists() {
            cmd.arg("-i").arg(path);
        } else {
            let dur = voice_durations[i];
            cmd.arg("-f").arg("lavfi").arg("-t").arg(dur.to_string()).arg("-i").arg("anullsrc=r=44100:cl=stereo");
        }
    }

    // 5. Build Filter Complex
    let mut filter = String::new();

    // Visual: Concat all 130 video segments
    for i in 0..130 {
        let mut fade = String::new();
        if i == 0 {
            // Studios: in 2s, out 3s at 4s (total 7s)
            fade = ",fade=t=in:st=0:d=2,fade=t=out:st=4:d=3".to_string();
        } else if i == 1 {
            // Title: in 2s, out 4s at 4s (total 8s)
            fade = ",fade=t=in:st=0:d=2,fade=t=out:st=4:d=4".to_string();
        }
        filter.push_str(&format!(
            "[{}:v]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p{}[v{}];",
            i, fade, i
        ));
    }
    for i in 0..130 { filter.push_str(&format!("[v{}]", i)); }
    filter.push_str("concat=n=130:v=1:a=0[vout];");

    // Audio: For each scene i, mix voice and SFX, and pad with silence for exfoliation scenes
    for i in 0..66 {
        let v_idx = 130 + i;
        let s_idx = 196 + i;
        
        // Prepare voice (silence for first two)
        if i < 2 {
            let dur = voice_durations[i];
            filter.push_str(&format!("anullsrc=r=44100:cl=stereo:d={}[vo{}];", dur, i));
        } else {
            filter.push_str(&format!("[{}:a]aresample=44100,aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[vo{}];", v_idx, i));
        }
        
        // Prepare sfx
        filter.push_str(&format!("[{}:a]aresample=44100,aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[sf{}];", s_idx, i));
        
        // Mix voice and sfx
        filter.push_str(&format!("[vo{}][sf{}]amix=inputs=2:duration=first:normalize=0[mix{}];", i, i, i));

        if i < 2 {
            // Intro: direct mix + pad
            let dur = voice_durations[i];
            filter.push_str(&format!("[mix{}]apad=whole_len={}[aseg{}];", i, (dur * 44100.0) as u32, i));
        } else {
            // Exfoliation: direct mix (plays during both images)
            filter.push_str(&format!("[mix{}]acopy[aseg{}];", i, i));
        }
    }

    
    // Concat all 66 audio segments
    for i in 0..66 { filter.push_str(&format!("[aseg{}]", i)); }
    filter.push_str("concat=n=66:v=0:a=1[aout];");

    cmd.arg("-filter_complex").arg(filter);
    cmd.arg("-map").arg("[vout]").arg("-map").arg("[aout]");
    
    if output_file.ends_with(".mp4") {
        cmd.arg("-c:v").arg("libx264").arg("-preset").arg("fast").arg("-crf").arg("18");
        cmd.arg("-c:a").arg("aac").arg("-b:a").arg("192k");
        cmd.arg("-pix_fmt").arg("yuv420p").arg("-r").arg(fps.to_string());
    } else {
        cmd.arg("-c:v").arg("mpeg2video").arg("-q:v").arg("2").arg("-b:v").arg("15000k");
        cmd.arg("-c:a").arg("mp2").arg("-b:a").arg("384k");
        cmd.arg("-pix_fmt").arg("yuv420p").arg("-r").arg(fps.to_string()).arg("-f").arg("mpeg");
    }
    cmd.arg(output_file);

    println!("Executing FFmpeg (Sync Logic)...");
    let status = cmd.status().expect("Failed to execute FFmpeg");
    if status.success() {
        println!("Successfully assembled: {}", output_file);
    } else {
        eprintln!("FFmpeg failed with status: {}", status);
    }
}
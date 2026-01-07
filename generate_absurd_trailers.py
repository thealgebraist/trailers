import torch
import scipy.io.wavfile
import os
import gc
import subprocess
import re
import requests
import numpy as np
import PIL.Image
if not hasattr(PIL.Image, "ANTIALIAS"): PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
from diffusers import FluxPipeline, StableAudioPipeline
from transformers import AutoProcessor, BarkModel

# --- Configuration ---
OUTPUT_DIR_PREFIX = "assets_"
if torch.cuda.is_available(): DEVICE = "cuda"
elif torch.backends.mps.is_available(): DEVICE = "mps"
else: DEVICE = "cpu"

# --- Movie Definitions ---
MOVIES = [
    {
        "id": "moistening",
        "title": "The Moistening",
        "vo": "In a world that was once crisp and dry, a strange phenomenon has begun. The walls are weeping, the bread is soggy, and the very air feels like a cold, wet sponge. This summer, prepare to feel slightly uncomfortable. No towel is large enough. No surface is safe. Experience the horror of a world that is just a little bit too damp. The Moistening.",
        "scenes": [
            "Photorealistic cinematic shot of a living room where the wallpaper is visibly damp and peeling in heavy strips. 8k.",
            "Photorealistic cinematic close up of a hand trying to pick up a slice of bread that is completely saturated and falling apart. 8k.",
            "Photorealistic cinematic shot of a sad wet cat sitting on a soggy velvet sofa, looking into the camera with despair. 8k.",
            "Photorealistic cinematic title card 'THE MOISTENING' written in water droplets on a fogged-up window pane. 8k."
        ]
    },
    {
        "id": "gavelgeddon",
        "title": "Gavel-geddon",
        "vo": "Court is in session, but the law has never been this delicious. Judge Miller has a reputation for being tough on crime, but even tougher on a rack of ribs. In a legal system where the jury is hungry and the lawyers are professional eaters, one man must defend his honor... and his appetite. Objection! Overruled by a side of slaw. Gavel-geddon.",
        "scenes": [
            "Photorealistic cinematic shot of a judge in black robes sitting at a high bench, aggressively eating a massive plate of messy barbecue ribs. 8k.",
            "Photorealistic cinematic shot of a courtroom where the lawyers are arguing while holding giant turkey legs. 8k.",
            "Photorealistic cinematic shot of a jury box where every juror is wearing a white lobster bib and holding a x. 8k.",
            "Photorealistic cinematic title card 'GAVEL-GEDDON' made of thick, bright yellow mustard on a mahogany table. 8k."
        ]
    },
    {
        "id": "sentient_scone",
        "title": "The Sentient Scone",
        "vo": "Arthur was just a lonely baker looking for a sign. He didn't expect the sign to come with raisins and a buttery attitude. From the moment he pulled it from the oven, he knew this scone was different. It didn't just smell good; it had opinions on his dating life. This Christmas, fall in love with something you can also eat. The Sentient Scone.",
        "scenes": [
            "Photorealistic cinematic shot of a glowing golden scone on a cooling rack. The scone appears to have a tiny, judgmental face formed by cracks and raisins. 8k.",
            "Photorealistic cinematic shot of a lonely baker in a flour-dusted apron whispering secrets to a scone on a plate. 8k.",
            "Photorealistic cinematic shot of a romantic candlelit dinner where a man is sitting across from a single scone on a velvet chair. 8k.",
            "Photorealistic cinematic title card 'THE SENTIENT SCONE' in elegant pastry script on a marble counter. 8k."
        ]
    },
    {
        "id": "tax_audit_musical",
        "title": "Tax Audit: The Musical",
        "vo": "The numbers don't add up, but the choreography is flawless. Agent Sterling is the IRS's top investigator, and he's got a rhythm that the law can't handle. When a simple filing error turns into a high-stakes jazz-tap sequence, the only thing higher than the interest is the high notes. Coming this April. You can't deduct the drama. Tax Audit: The Musical.",
        "scenes": [
            "Photorealistic cinematic shot of a stern IRS agent in a suit, performing a mid-air leap in a grey office filled with flying tax forms. 8k.",
            "Photorealistic cinematic shot of a row of accountants in green visors doing a synchronized dance with calculators. 8k.",
            "Photorealistic cinematic shot of a man screaming in terror at a mountain of receipts while a spotlight hits him. 8k.",
            "Photorealistic cinematic title card 'TAX AUDIT: THE MUSICAL' in neon lights inside a dark file storage room. 8k."
        ]
    },
    {
        "id": "beige_alert",
        "title": "Beige Alert",
        "vo": "The world was a vibrant tapestry of color, until the saturation started to fail. First went the blues, then the reds, leaving behind a terrifying expanse of oatmeal, sand, and taupe. As the global panic sets in, one man must find the last remaining tube of neon pink paint before the entire planet becomes a neutral-toned nightmare. Beige Alert.",
        "scenes": [
            "Photorealistic cinematic wide shot of a city street where everything—cars, trees, people—is exactly the same shade of dull beige. 8k.",
            "Photorealistic cinematic shot of a sunset that is entirely composed of different shades of tan and light brown. 8k.",
            "Photorealistic cinematic shot of a man holding a single, vibrant glowing pink paint tube in a world of oatmeal colors. 8k.",
            "Photorealistic cinematic title card 'BEIGE ALERT' in a very slightly darker beige font on a beige background. 8k."
        ]
    },
    {
        "id": "stapler_heist",
        "title": "The Great Stapler Heist",
        "vo": "It's the ultimate office tool. A Swingline 747 in limited edition chrome. And it's currently held in the highest security cubicle in the world. Ten floors. Fifty guards. One very determined temp. This summer, the paper remains unfastened until the job is done. Stick to the plan. Don't jam the mechanism. The Great Stapler Heist.",
        "scenes": [
            "Photorealistic cinematic shot of a shiny chrome stapler sitting inside a laser-protected glass display case in a corporate office. 8k.",
            "Photorealistic cinematic shot of a man in a black tactical outfit crawling through an office ceiling vent. 8k.",
            "Photorealistic cinematic shot of a team of office workers using a coffee machine to create a distraction. 8k.",
            "Photorealistic cinematic title card 'THE GREAT STAPLER HEIST' embossed on a giant metal staple. 8k."
        ]
    },
    {
        "id": "lintpocalypse",
        "title": "Lint-pocalypse",
        "vo": "It started in a laundromat in Queens. A small ball of fluff that everyone ignored. But it grew. It consumed socks, it consumed sweaters, and now, it's consuming the city. Soft, grey, and utterly unstoppable. The more you try to clean it, the stronger it gets. This July, the dryer is empty, and the world is fuzzy. Lint-pocalypse.",
        "scenes": [
            "Photorealistic cinematic shot of a giant, 50-foot tall ball of dryer lint rolling through a city street, picking up cars. 8k.",
            "Photorealistic cinematic shot of a terrified scientist looking at a microscopic piece of lint under a glowing blue laser. 8k.",
            "Photorealistic cinematic shot of a city covered in a thick, grey, fuzzy layer of lint as if it were snow. 8k.",
            "Photorealistic cinematic title card 'LINTPOCALYPSE' written in fluffy, fuzzy letters on a dark grey background. 8k."
        ]
    },
    {
        "id": "puddle_jumper",
        "title": "Puddle Jumper",
        "vo": "He doesn't fly. He isn't strong. But his shoes are always, always dry. In a city that never stops raining, one man has the power to leap over any obstacle, provided that obstacle is a small body of standing water. When the villain 'The Soaker' threatens to flood the city, only one man can jump to the rescue. Puddle Jumper. He's got great form.",
        "scenes": [
            "Photorealistic cinematic shot of a man in a raincoat performing a majestic, slow-motion leap over a large city puddle. 8k.",
            "Photorealistic cinematic shot of a villain in a diving suit pointing a giant fire hose at a crowd. 8k.",
            "Photorealistic cinematic shot of a pair of pristine, perfectly dry white sneakers standing in the middle of a muddy street. 8k.",
            "Photorealistic cinematic title card 'PUDDLE JUMPER' in clean, waterproof blue letters. 8k."
        ]
    },
    {
        "id": "cardboard_chronicles",
        "title": "The Chronicles of Cardboard",
        "vo": "In the kingdom of Corrugatia, everything is built to last... as long as it doesn't rain. Prince Flatpack must lead his paper-thin army against the rising damp of the north. One drop could destroy their heritage. One storm could end their world. This fantasy epic is paper-thin, but the stakes are high. Fold your destiny. The Chronicles of Cardboard.",
        "scenes": [
            "Photorealistic cinematic shot of a majestic castle made entirely of brown corrugated cardboard, complete with cardboard flags. 8k.",
            "Photorealistic cinematic shot of a knight in cardboard armor riding a cardboard horse across a flat paper plain. 8k.",
            "Photorealistic cinematic shot of a single giant raindrop falling toward a cardboard city in slow motion. 8k.",
            "Photorealistic cinematic title card 'THE CHRONICLES OF CARDBOARD' cut out of a rough shipping box. 8k."
        ]
    },
    {
        "id": "hover_grandma",
        "title": "Hover-Boarding with Grandma",
        "vo": "She's 92 years old. She's got a ceramic hip. And she's the most dangerous hoverboarder in the underground circuit. They told her to stay in the knitting circle, but Grandma has a Need for Speed and a literal anti-gravity drive. This summer, the senior center is going mobile. Watch out for the dentures. Hover-Boarding with Grandma.",
        "scenes": [
            "Photorealistic cinematic shot of an elderly woman with white hair and a floral dress, performing a kickflip on a glowing neon hoverboard. 8k.",
            "Photorealistic cinematic shot of a high-speed hoverboard chase through a retirement home hallway. 8k.",
            "Photorealistic cinematic shot of Grandma putting on a futuristic chrome helmet over her glasses. 8k.",
            "Photorealistic cinematic title card 'HOVER-BOARDING WITH GRANDMA' in bright neon pink and teal. 8k."
        ]
    },
    {
        "id": "mildly_inconvenienced",
        "title": "The Mildly Inconvenienced",
        "vo": "The apocalypse wasn't a bang. It was a spinning loading icon. In a world where the infrastructure has collapsed to the point where every internet connection has a five-second lag, society is on the brink of total annoyance. One man must trek across the wasteland to find the last working ethernet cable. It's going to be a very frustrating journey. The Mildly Inconvenienced.",
        "scenes": [
            "Photorealistic cinematic shot of a post-apocalyptic survivor staring in absolute fury at a smartphone showing a 99% loading bar. 8k.",
            "Photorealistic cinematic shot of a wasteland made of old routers and tangled computer cables. 8k.",
            "Photorealistic cinematic shot of a man trying to open a door that is slightly stuck, looking incredibly annoyed. 8k.",
            "Photorealistic cinematic title card 'THE MILDLY INCONVENIENCED' with a spinning loading circle next to it. 8k."
        ]
    },
    {
        "id": "missing_sock",
        "title": "The Mystery of the Missing Left Sock",
        "vo": "Detective Malone thought he had seen it all. But then he looked in his dryer. Seven right socks. Zero lefts. It's a conspiracy that goes all the way to the top of the hamper. In a city where the laundry is dirty and the mysteries are wool-blend, one man will stop at nothing to find the matching pair. This case is full of holes. The Mystery of the Missing Left Sock.",
        "scenes": [
            "Photorealistic cinematic shot of a gritty detective in a trench coat, holding a single magnifying glass over a washing machine. 8k.",
            "Photorealistic cinematic shot of a dark alleyway paved with thousands of single, unmatched socks. 8k.",
            "Photorealistic cinematic shot of a sinister man in a laundry room, hiding a bag full of left socks. 8k.",
            "Photorealistic cinematic title card 'THE MYSTERY OF THE MISSING LEFT SOCK' printed on a laundry tag. 8k."
        ]
    },
    {
        "id": "50_foot_toddler",
        "title": "Attack of the 50-Foot Toddler",
        "vo": "He wants his bottle. He wants his nap. And he wants to stomp on the Chrysler Building. When a growth spurt goes horribly wrong, the city becomes a giant playroom. The military has tanks, but he has a very high-pitched scream and no concept of personal space. This summer, the big baby is in town. Attack of the 50-Foot Toddler.",
        "scenes": [
            "Photorealistic cinematic shot of a giant toddler sitting in the middle of a city intersection, holding a bus like a toy car. 8k.",
            "Photorealistic cinematic shot of fighter jets flying around a giant baby's head as it giggles. 8k.",
            "Photorealistic cinematic shot of a 50-foot tall pacifier lying in the middle of a destroyed park. 8k.",
            "Photorealistic cinematic title card 'ATTACK OF THE 50-FOOT TODDLER' written in giant colorful toy blocks. 8k."
        ]
    },
    {
        "id": "accountant_arthur",
        "title": "The Accountant of the Round Table",
        "vo": "Pulling the sword from the stone was the easy part. The real challenge was the quarterly expense reports. While the knights are out slaying dragons, one man is back at Camelot, making sure the dragon-slaying budget is strictly followed. In a world of magic and monsters, the most powerful weapon is a well-organized spreadsheet. The Accountant of the Round Table.",
        "scenes": [
            "Photorealistic cinematic shot of a medieval scribe sitting at a desk next to King Arthur, holding a quill and an abacus. 8k.",
            "Photorealistic cinematic shot of a Round Table covered in scrolls, ink pots, and tax documents. 8k.",
            "Photorealistic cinematic shot of a dragon being presented with a bill for 'property damage'. 8k.",
            "Photorealistic cinematic title card 'THE ACCOUNTANT OF THE ROUND TABLE' in gothic calligraphy on parchment. 8k."
        ]
    },
    {
        "id": "elevator_pitch",
        "title": "Elevator Pitch",
        "vo": "It's the movie about the movie about the movie. Set entirely within a thirty-second elevator ride, three producers must decide the fate of cinema. Every floor is a new genre. Every button is a plot twist. It's a high-concept thriller with very low ceilings. Going up? Elevator Pitch. It's an uplifting story.",
        "scenes": [
            "Photorealistic cinematic shot of three men in suits standing in a cramped, mirrored elevator, looking intensely at each other. 8k.",
            "Photorealistic cinematic shot of the elevator floor indicator lights glowing with strange symbols. 8k.",
            "Photorealistic cinematic shot of an elevator door opening to reveal a jungle, then closing immediately. 8k.",
            "Photorealistic cinematic title card 'ELEVATOR PITCH' written on a small 'Out of Order' sign. 8k."
        ]
    },
    {
        "id": "bean_man",
        "title": "The Unbearable Lightness of Beans",
        "vo": "Harold was a simple man until the incident at the pantry. Now, he's convinced that he is slowly, inevitably, turning into a pinto bean. His friends don't understand. His doctor is confused. But Harold knows the truth: the fiber is calling. A philosophical drama about identity, destiny, and legumes. The Unbearable Lightness of Beans.",
        "scenes": [
            "Photorealistic cinematic shot of a man staring intensely at a single dried bean on a white pedestal. 8k.",
            "Photorealistic cinematic shot of a man's skin beginning to take on the texture and color of a brown bean. 8k.",
            "Photorealistic cinematic shot of a man sitting in a bathtub filled with thousands of baked beans, looking thoughtful. 8k.",
            "Photorealistic cinematic title card 'THE UNBEARABLE LIGHTNESS OF BEANS' in a minimalist, high-brow font. 8k."
        ]
    }
]

# --- Helper Functions ---
def flush():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def apply_trailer_voice_effect(file_path):
    temp_path = file_path.replace(".wav", "_temp.wav")
    filter_complex = "lowshelf=g=15:f=100,highshelf=g=-5:f=8000,acompressor=threshold=-12dB:ratio=4:makeup=4dB"
    cmd = ["ffmpeg", "-y", "-i", file_path, "-af", filter_complex, temp_path]
    try: subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); os.replace(temp_path, file_path)
    except Exception as e: print(f"Failed effect: {e}")

def generate_voice(movie):
    out_dir = f"assets_{movie['id']}/voice"
    os.makedirs(out_dir, exist_ok=True)
    output_path = f"{out_dir}/voiceover_full.wav"
    
    if os.path.exists(output_path) and "voice" not in sys.argv:
        return

    print(f"Generating Bark VO for: {movie['title']}")
    try:
        processor = AutoProcessor.from_pretrained("suno/bark")
        model = BarkModel.from_pretrained("suno/bark", torch_dtype=torch.float32).to(DEVICE)
        voice_preset = "v2/en_speaker_9"
        sample_rate = model.generation_config.sample_rate
        
        # Split text into sentences to avoid Bark hallucinations/fillers
        sentences = [s.strip() for s in movie['vo'].split(".") if s.strip()]
        full_audio = []
        for sent in sentences:
            clean_text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?]', '', sent)
            inputs = processor(clean_text, voice_preset=voice_preset, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                audio_array = model.generate(**inputs, min_eos_p=0.05).cpu().numpy().squeeze()
            full_audio.append(audio_array)
            full_audio.append(np.zeros(int(0.2 * sample_rate), dtype=np.float32)) # Minimal gap
            
        combined = np.concatenate(full_audio)
        scipy.io.wavfile.write(output_path, rate=sample_rate, data=combined)
        apply_trailer_voice_effect(output_path)
        del model; del processor; flush()
    except Exception as e: print(f"Bark failed: {e}")

def generate_images(movie):
    img_dir = f"assets_{movie['id']}/images"
    os.makedirs(img_dir, exist_ok=True)
    
    print(f"Generating Images for: {movie['title']}")
    model_id = "black-forest-labs/FLUX.1-schnell"
    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    if DEVICE == "cuda": pipe.enable_model_cpu_offload(); pipe.enable_vae_tiling()
    else: pipe.to(DEVICE)
    
    for i, prompt in enumerate(movie['scenes']):
        filename = f"{img_dir}/clip_{i}.png"
        txt_filename = f"{img_dir}/clip_{i}.txt"
        if os.path.exists(filename): continue
        
        image = pipe(prompt=prompt, guidance_scale=0.0, num_inference_steps=8, max_sequence_length=256).images[0]
        image.save(filename)
        with open(txt_filename, "w") as f: f.write(f"Prompt: {prompt}\nSteps: 8\n")
    
    del pipe; flush()

def generate_audio(movie):
    music_dir = f"assets_{movie['id']}/music"
    os.makedirs(music_dir, exist_ok=True)
    
    print(f"Generating Music for: {movie['title']}")
    pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float32)
    if DEVICE == "cuda": pipe.enable_model_cpu_offload()
    else: pipe.to(DEVICE)
    
    m_path = f"{music_dir}/theme.wav"
    if not os.path.exists(m_path):
        audio = pipe(prompt=f"Cinematic trailer music for a movie titled {movie['title']}", num_inference_steps=100, audio_end_in_s=45.0).audios[0]
        scipy.io.wavfile.write(m_path, rate=44100, data=audio.cpu().numpy().T)
        
    del pipe; flush()

if __name__ == "__main__":
    import sys
    for m in MOVIES:
        if "voice" in sys.argv:
            generate_voice(m)
        else:
            generate_voice(m)
            generate_images(m)
            generate_audio(m)
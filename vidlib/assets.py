# vidlib.assets: Asset generation functions
from ..generate_chimp_assets import generate_images as chimp_generate_images, generate_sfx as chimp_generate_sfx, generate_voiceover as chimp_generate_voiceover, generate_music as chimp_generate_music
from ..generate_dalek_assets import generate_images as dalek_generate_images, generate_sfx as dalek_generate_sfx, generate_voiceover as dalek_generate_voiceover, generate_music as dalek_generate_music
from ..generate_metro_assets import generate_images as metro_generate_images, generate_sfx as metro_generate_sfx, generate_voiceover as metro_generate_voiceover, generate_music as metro_generate_music, apply_trailer_voice_effect
from ..generate_sloppy_assets import generate_images as sloppy_generate_images, generate_sfx as sloppy_generate_sfx, generate_voiceover as sloppy_generate_voiceover, generate_music as sloppy_generate_music

__all__ = [
    'chimp_generate_images', 'chimp_generate_sfx', 'chimp_generate_voiceover', 'chimp_generate_music',
    'dalek_generate_images', 'dalek_generate_sfx', 'dalek_generate_voiceover', 'dalek_generate_music',
    'metro_generate_images', 'metro_generate_sfx', 'metro_generate_voiceover', 'metro_generate_music', 'apply_trailer_voice_effect',
    'sloppy_generate_images', 'sloppy_generate_sfx', 'sloppy_generate_voiceover', 'sloppy_generate_music',
]

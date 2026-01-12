from transformers import MusicgenConfig, AutoConfig
model_id = "facebook/musicgen-small"
config = AutoConfig.from_pretrained(model_id)
print(f"Config type: {type(config)}")
print(f"Has decoder: {hasattr(config, 'decoder')}")
if hasattr(config, 'decoder'):
    print(f"Decoder type: {type(config.decoder)}")

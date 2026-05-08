# Color palettes used across plots: layer-combination colors (3-bit indexed) and a pastel mixer.

COLOR_MAP3 = Dict(
    0b000 => (0,0,0,0),      # White (none)
    0b001 => (0,0,1,1),      # Blue (w3)
    0b010 => (0,1,0,1),      # Green (w2)
    0b011 => (0,1,1,1),      # Cyan (w2 & w3)
    0b100 => (1,0,0,1),      # Red (w1)
    0b101 => (1,0,1,1),      # Magenta (w1 & w3)
    0b110 => (1,0.7,0,1),    # Orange (w1 & w2) - changed from (1,1,0,1) yellow
    0b111 => (0,0,0,1)       # all (black)
)

function to_pastel(color)
    white = (1.0, 1.0, 1.0, 1.0)
    mix_ratio = 0.7  # Adjust this ratio to get the desired pastel effect
    return (
        mix_ratio * color[1] + (1 - mix_ratio) * white[1],
        mix_ratio * color[2] + (1 - mix_ratio) * white[2],
        mix_ratio * color[3] + (1 - mix_ratio) * white[3],
        color[4]  # Keep the alpha value unchanged
    )
end

COLOR_MAP3_PASTEL = Dict(k => to_pastel(v) for (k, v) in COLOR_MAP3)


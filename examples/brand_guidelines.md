# Crisis Corps Brand Guidelines for AI Generation

## Logo Design Principles

### Core Visual Elements
1. **Robot Symbol**: Heroic, protective stance
2. **Shield/Badge Shape**: Conveys protection and authority  
3. **Typography**: Bold, sans-serif, military/tech inspired
4. **Geometric Construction**: Clean lines, scalable design

### Color Palette

**Primary Colors:**
- **Emergency Orange**: #FF6B35 (RGB: 255, 107, 53)
- **Tech Blue**: #004E98 (RGB: 0, 78, 152)
- **White**: #FFFFFF (Background/negative space)

**Secondary Colors:**
- **Steel Grey**: #4A5568 (Accents)
- **Success Green**: #48BB78 (Achievements)
- **Alert Red**: #E53E3E (Warnings)

### Prompt Engineering Tips

#### Positive Prompt Components
```
"Crisis Corps logo, [specific element], [color scheme], [style modifier], [technical specs]"
```

**Elements to Include:**
- minimalist design
- vector art style
- clean lines
- geometric shapes
- professional branding
- scalable graphics
- white background
- centered composition

#### Negative Prompt Components
```
"photograph, realistic, 3d render, gradient, shadow, complex, busy, cluttered, photorealistic"
```

**Avoid:**
- Photorealistic textures
- Complex gradients
- Soft/organic shapes
- Hand-drawn appearance
- Vintage/distressed effects

### Logo Variations Needed

1. **Primary Logo** (1024x1024)
   - Full logo with text and symbol
   - Use for: Website header, business cards

2. **Icon Only** (512x512)
   - Robot symbol without text
   - Use for: App icon, favicon

3. **Horizontal Logo** (1920x480)
   - Wide format with text beside symbol
   - Use for: Email signatures, banners

4. **Monogram** (256x256)
   - "CC" stylized letters
   - Use for: Social media profile pictures

5. **Badge/Emblem** (768x768)
   - Military-style insignia
   - Use for: Game achievements, pins

### ComfyUI Workflow Settings

**Recommended Parameters:**
- **Model**: SD XL Base 1.0 or similar
- **Steps**: 30-40 for final quality
- **CFG Scale**: 8-11 (higher for more prompt adherence)
- **Sampler**: DPM++ 2M Karras or DPM++ 3M SDE
- **Batch Size**: 4-6 for variations

### Style References

**Similar Brands to Reference:**
- Overwatch (hero shooter game)
- XCOM (tactical military)
- Boston Dynamics (robotics)
- SpaceX (tech/future)
- Red Cross (humanitarian)

### Usage Examples

```python
# Generate main logo
await generate_image(
    prompt="Crisis Corps logo, heroic robot head in hexagonal shield, "
           "emergency orange and tech blue, minimalist vector design, "
           "white background, professional branding, scalable graphics",
    negative_prompt="realistic, photograph, 3d render, gradient shadows",
    width=1024,
    height=1024,
    cfg_scale=10.0,
    steps=35
)

# Generate app icon
await generate_image(
    prompt="Crisis Corps app icon, simplified robot face, rounded square, "
           "flat design, orange accent lights on blue background, "
           "mobile app store ready, minimalist, memorable",
    width=512,
    height=512,
    cfg_scale=11.0,
    steps=30
)
```

### Testing Checklist

- [ ] Readable at small sizes (32x32px)
- [ ] Works in monochrome
- [ ] Clear silhouette
- [ ] Maintains identity without color
- [ ] Scalable to billboard size
- [ ] Recognizable from distance
- [ ] Unique among competitors
- [ ] Conveys both tech and humanitarian aspects

### File Naming Convention

```
crisis_corps_[type]_[variation]_[size]_[version].png

Examples:
- crisis_corps_logo_main_1024_v1.png
- crisis_corps_icon_app_512_v2.png  
- crisis_corps_emblem_gold_768_v1.png
```

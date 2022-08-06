from wand.image import Image as wImage
from wand.display import display as wdiplay
from wand.drawing import Drawing
from wand.color import Color
import arabic_reshaper
from bidi.algorithm import get_display

reshaped_text = 'واژهٓ كَزَعْمِ عریانیده‌ای بخشش‌طلبی بی،۱ماران توسکلانی سود،مرگ محرابگه خلفیش ٱرْحَمْهُمَا انگولا شد،فريادش بهرام‌شاه میرزااحمد‌خان بهره‌می‌برد'
artext = get_display(reshaped_text)
print(reshaped_text)
fonts = ['/home/create-data/fonts/IranNastaliq.ttf']
draw = Drawing()
img =  wImage(width=1200,height=(len(fonts)+2)*60,background=Color('#ffffff')) 
#draw.fill_color(Color('#000000'))
draw.text_alignment = 'right'
draw.text_antialias = True
draw.text_encoding = 'utf-8'
#draw.text_interline_spacing = 1
#draw.text_interword_spacing = 15.0
draw.text_kerning = 0.0
for i in range(len(fonts)):
    font =  fonts[i]
    draw.font = font
    draw.font_size = 40
    draw.text(int(img.width / 2), 40+(i*60),artext)
    print(draw.get_font_metrics(img,artext))
    draw(img)
draw.text(int(img.width / 2), 40+((i+1)*60),u'ناصر test')
draw(img)
img.save(filename='out.jpg')
wdiplay(img)
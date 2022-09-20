import gdown

def download_gdrive(key, output):
    url = f"https://drive.google.com/uc?id={key}"
    gdown.download(url, output, quiet=False)

KEYS = {
    "derain" : ['1uuejKpyo0G_5M4DAO2J9_Dijy550tjc5', 'derain'],
    "denoise" : ["1FF_4NTboTWQ7sHCq4xhyLZsSl0U0JfjH", 'denoise'], 
    "motion_deblur" : ["1pwcOhDS5Erzk8yfAbu7pXTud606SB4-L", 'motion_deblur'],
    "single_image_defocus_deblurring" : ["10v8BH3Gktl34TYzPy0x-pAKoRSYKnNZp", 'defocus_deblur'],
    "dual_pixel_defocus_deblurring" : ["167enijHIBa1axZRaRjkk_U6kLKm40Z43", 'defocus_deblur']
}

def main():    
    for k, v in KEYS.items():
        download_gdrive(v[0], f"./configs/{v[1]}/{k}.pth"  )

if __name__ == "__main__":
    main()

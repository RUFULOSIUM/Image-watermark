import image_watermark.encode as encode
import image_watermark.decode as decode

def main():
    encode.encode_image("./images/test.png", "./encodet/test.png", "Refrlections and refractions from glass objects", "myself", "v0.2")
    decode.decode_image("./encodet/test.png")


if __name__ == "__main__":
    main()

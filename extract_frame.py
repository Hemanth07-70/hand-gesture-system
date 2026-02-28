with open("test_feed.mjpeg", "rb") as f:
    data = f.read()
    start = data.find(b"\xff\xd8")
    if start != -1:
        end = data.find(b"\xff\xd9", start)
        if end != -1:
            with open("static/extracted_frame.jpg", "wb") as out:
                out.write(data[start:end+2])
            print("Successfully extracted frame to static/extracted_frame.jpg")
        else:
            print("Could not find end of JPEG")
    else:
        print("Could not find start of JPEG")

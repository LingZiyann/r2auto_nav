import numpy as np 
import cv2 

def detect_color_and_draw():
    # Capturing video through webcam 
    webcam = cv2.VideoCapture(0) 

    detected_color = "none"
    output_size = (680, 480)  # Example: resize to 640x480

    # Define the color ranges and drawing colors
    color_ranges = {
        "red": ([136, 87, 111], [180, 255, 255], (0, 0, 255)),  # BGR for red
        "green": ([25, 52, 72], [102, 255, 255], (0, 255, 0)),  # BGR for green
        "blue": ([94, 80, 2], [120, 255, 255], (255, 0, 0))     # BGR for blue
    }

    while True:
        _, imageFrame = webcam.read() 
        imageFrame = cv2.resize(imageFrame, output_size)
        hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV) 
        max_area = 0


        # Iterate through the color ranges
        for color, (lower, upper, draw_color) in color_ranges.items():
            lower = np.array(lower, np.uint8)
            upper = np.array(upper, np.uint8)
            mask = cv2.inRange(hsvFrame, lower, upper)
            mask = cv2.dilate(mask, np.ones((5, 5), "uint8"))

            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 800:
                    x, y, w, h = cv2.boundingRect(contour)
                    imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), draw_color, 2)
                    cv2.putText(imageFrame, f"{color} color", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, draw_color)
                    
                    # Update detected_color and max_area if this contour is the largest
                    if area > max_area:
                        max_area = area
                        detected_color = color
                        print(detected_color)


        cv2.imshow("Multiple Color Detection in Real-Time", imageFrame) 
        if cv2.waitKey(10) & 0xFF == ord('q'): 
            break

    webcam.release() 
    cv2.destroyAllWindows() 
    return detected_color

# Example usage
if __name__ == "__main__":
    detected_color = detect_color_and_draw()
    print(f"Detected color: {detected_color}")

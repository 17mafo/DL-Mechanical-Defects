import cv2
import numpy as np

class ImagePreprocessor:
    def __init__(self, img_path):
        # Segment, based on range of green --> make into mask, largest section selected
        self.lower_green = np.array([35, 40, 40])
        self.upper_green = np.array([85, 255, 255])
        self.img = cv2.imread(img_path)
        self.hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        self.hole_mask = cv2.inRange(self.hsv, self.lower_green, self.upper_green)
        kernel = np.ones((5,5), np.uint8)
        self.hole_mask = cv2.morphologyEx(self.hole_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        self.hole_mask = cv2.morphologyEx(self.hole_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        # cv2.imshow("hole_mask", self.hole_mask)
        # cv2.waitKey(0)

    def image_initial_cutting(self,brim_px=11):

        # lower_orange = np.array([5, 40, 40])
        # upper_orange = np.array([30, 255, 255])
        # hole_mask = cv2.inRange(hsv, lower_orange, upper_orange)

        # Make find largest to contour and make circle, take radious and add on artifical amount to take out correct
        contours, _ = cv2.findContours(self.hole_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hole_cnt = max(contours, key=cv2.contourArea)
        (cx, cy), hole_radius = cv2.minEnclosingCircle(hole_cnt)
        cx, cy, hole_radius = int(cx), int(cy), int(hole_radius)
        brim_radius = hole_radius + brim_px
        h, w = self.img.shape[:2]
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        keep_mask = dist <= brim_radius

        # Make everything outside the keep_mask white
        out = self.img.copy()
        out[~keep_mask] = 255

        # Crop image to bounding box of mask
        ys, xs = np.where(keep_mask)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()

        out_cropped = out[y0:y1+1, x0:x1+1]

        return out_cropped
    
    def outer_rim_cutting(self, rim_px=11):
        return None
    
    def background_area(self):
        return None

# Example
hole_plus_metal = ImagePreprocessor.image_initial_cutting("C:\\Users\\marti\\Documents\\DL-Mechanical-Defects\\dataset_creation\\images\\bad\\3_bad_focus_1.jpg")
cv2.imshow("hole_plus_metal.png", hole_plus_metal)
cv2.waitKey(0)
cv2.destroyAllWindows()
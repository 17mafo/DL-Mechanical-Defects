import cv2
import numpy as np

class ImagePreprocessor:
    def __init__(self, img_path):
  # Load image and convert to HSV
        self.img = cv2.imread(img_path)
        self.hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

        # Target green color in RGB (from RGBA)
        rgb_target = np.array([18, 78, 6], dtype=np.uint8).reshape(1, 1, 3)
        hsv_target = cv2.cvtColor(rgb_target, cv2.COLOR_RGB2HSV)[0,0]
        h, s, v = int(hsv_target[0]), int(hsv_target[1]), int(hsv_target[2])

        # Create a tight HSV range around target to remove dark/dull areas
        self.lower_green = np.array([max(h-10,0), 150, 50])  # limit S>150, V>50
        self.upper_green = np.array([min(h+10,179), 255, 255])

        # Mask green areas
        self.hole_mask = cv2.inRange(self.hsv, self.lower_green, self.upper_green)

        # Clean mask using morphology
        kernel = np.ones((5,5), np.uint8)
        self.hole_mask = cv2.morphologyEx(self.hole_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        self.hole_mask = cv2.morphologyEx(self.hole_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        self.green_cutout_mask = None  # to be defined in methods

    def image_initial_cutting(self,brim_px=20):

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
        self.green_cutout_mask = dist <= brim_radius

        # Make everything outside the green_cutout_mask white
        out = self.img.copy()
        out[~self.green_cutout_mask] = 255
        # Crop image to bounding box of mask
        ys, xs = np.where(self.green_cutout_mask)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()

        out_cropped = out[y0:y1+1, x0:x1+1]

        return out_cropped
    
    def outer_rim_cutting(self, brim_px=20, rim_px=11):
        """
        Return only the outer rim: part of initial cutting minus background green area.
        brim_px: how much extra radius to include in initial cutting
        rim_px: thickness of outer rim to keep
        """
        # Get initial cut (hole + metal/brim)
        initial_cut = self.image_initial_cutting(brim_px=brim_px)

        # Get green background area
        background = self.background_area()
        if background is None:
            return initial_cut  # fallback if no green detected

        # Resize background to match initial_cut size (in case crop sizes differ)
        h0, w0 = initial_cut.shape[:2]
        h1, w1 = background.shape[:2]
        if (h0, w0) != (h1, w1):
            # We need to place background mask in same coordinate frame
            # First create a blank image of initial_cut size
            bg_mask = np.ones((h0, w0), dtype=np.uint8) * 255  # white
            # Determine where to place background in this blank image
            # For simplicity, resize background to initial_cut size
            background_resized = cv2.resize(background, (w0, h0))
        else:
            background_resized = background

        # Create masks: white areas = background/keep, black = outer rim
        # Make mask of background area (green part)
        bg_gray = cv2.cvtColor(background_resized, cv2.COLOR_BGR2GRAY)
        _, bg_mask = cv2.threshold(bg_gray, 254, 255, cv2.THRESH_BINARY_INV)  # green area = 255

        # Create mask for outer rim: subtract green area from initial_cut circle
        initial_gray = cv2.cvtColor(initial_cut, cv2.COLOR_BGR2GRAY)
        _, initial_mask = cv2.threshold(initial_gray, 254, 255, cv2.THRESH_BINARY_INV)

        # Outer rim = initial mask minus green area mask
        outer_rim_mask = cv2.subtract(initial_mask, bg_mask)

        # Apply mask to initial_cut image
        outer_rim = initial_cut.copy()
        outer_rim[outer_rim_mask == 0] = 255  # make non-rim white

        return outer_rim


    
    def background_area(self):
        # Find all contours in the green mask
        contours, _ = cv2.findContours(self.hole_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None  # no green detected

        # Keep only the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Create a mask for just the largest green area
        largest_mask = np.zeros_like(self.hole_mask)
        cv2.drawContours(largest_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        # Apply mask to image
        green_only = self.img.copy()
        green_only[largest_mask == 0] = 255  # set non-green to white

        # Crop to bounding box of largest green area
        ys, xs = np.where(self.green_cutout_mask > 0)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        green_only = green_only[y0:y1+1, x0:x1+1]

        # Display
        cv2.imshow("Largest Green Area Only", green_only)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return green_only



# Example
imageproce = ImagePreprocessor("C:\\Users\\marti\\Documents\\DL-Mechanical-Defects\\dataset_creation\\images\\bad\\3_bad_focus_2.jpg")
imageproce.outer_rim_cutting()
cv2.imshow("outer_rim.png", imageproce.outer_rim_cutting())


hole_plus_metal = imageproce.image_initial_cutting()
cv2.imshow("hole_plus_metal.png", hole_plus_metal)
cv2.waitKey(0)
cv2.destroyAllWindows()
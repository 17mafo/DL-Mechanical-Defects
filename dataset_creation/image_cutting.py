import cv2
import numpy as np

class ImagePreprocessor:
    def __init__(self, img_path):
        self.img = cv2.imread(img_path)

        # Mask green areas
        self.hole_mask = cv2.inRange(self.img, np.array([0, 65, 0]), np.array([50, 250, 80]))
        
        self.imgcontrast = cv2.convertScaleAbs(self.img, alpha=3, beta=-20)
        
        cv2.imshow("contrast", self.imgcontrast)
        self.region_mask = self.test_floodfill_from_green((40,40,40), (90,210,90), display=False)
        # Find contours of green areas
        self.contours, _ = cv2.findContours(self.region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not self.contours:
            raise Exception("No contours found in hole mask")

        self.largest_contour = max(self.contours, key=cv2.contourArea)

        self.contours_old, _ = cv2.findContours(self.hole_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not self.contours_old:
            raise Exception("No contours found in hole mask")

        self.largest_contour_old = max(self.contours_old, key=cv2.contourArea)
                
        # Clean mask (removes "worms" in green area)
        # kernel = np.ones((5,5), np.uint8)
        # self.hole_mask = cv2.morphologyEx(self.hole_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        # self.hole_mask = cv2.morphologyEx(self.hole_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # clean region_mask
        # self.region_mask = cv2.morphologyEx(self.region_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        # self.region_mask = cv2.morphologyEx(self.region_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        self.green_cutout_mask = None

    def image_initial_cutting(self, brim_px=20, display=False):
        # Make find largest to contour and make circle, take radious and add on artifical amount to take out correct
        (cx, cy), hole_radius = cv2.minEnclosingCircle(self.largest_contour_old)
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

        if display:
            cv2.imshow("Initial Cut (Hole + Metal/Brim)", out_cropped)

        return out_cropped
    
    def outer_rim_cutting(self, brim_px=20, display=False):

        # Get initial cut
        
        initial_cut = self.image_initial_cutting(brim_px=brim_px, display=display)
        # Get green background area
        background = self.background_area(display=display)

        # Create masks, white are in green area
        bg_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        _, bg_mask = cv2.threshold(bg_gray, 254, 255, cv2.THRESH_BINARY_INV)

        # Create mask for outer rim: subtract green area from initial_cut circle
        initial_gray = cv2.cvtColor(initial_cut, cv2.COLOR_BGR2GRAY)
        _, initial_mask = cv2.threshold(initial_gray, 254, 255, cv2.THRESH_BINARY_INV)

        # Outer rim = initial mask minus green area mask
        outer_rim_mask = cv2.subtract(initial_mask, bg_mask)

        # Apply mask to initial_cut image
        outer_rim = initial_cut.copy()
        outer_rim[outer_rim_mask == 0] = 255  # make non-rim white

        if display:
            cv2.imshow("Outer Rim Cut", outer_rim)

        return outer_rim
    
    def background_area(self, display=False):
        # Create a mask for just the largest green area
        largest_mask = np.zeros_like(self.region_mask)
        cv2.drawContours(largest_mask, [self.largest_contour], -1, 255, thickness=cv2.FILLED)

        # Apply mask to image and set non-green to white
        green_only = self.img.copy()
        green_only[largest_mask == 0] = 255

        # Crop to bounding box of largest green area
        ys, xs = np.where(self.green_cutout_mask > 0)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        green_only = green_only[y0:y1+1, x0:x1+1]

        # Display
        if display:
            cv2.imshow("Largest Green Area Only", green_only)

        return green_only

    def test_floodfill_from_green(self, lo=10, up=10, display=True):

        # Find seed point as center of green mask
        ys, xs = np.where(self.hole_mask > 0)
        if len(xs) == 0:
            raise Exception("hole_mask is empty")

        seed_x = int(xs.mean())
        seed_y = int(ys.mean())
        seed = (seed_x, seed_y)

        # Prepare floodFill mask (must be 2 pixels larger than image)
        h, w = self.img.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)

        # Use FIXED_RANGE so threshold is relative to seed pixel
        flags = 8 | cv2.FLOODFILL_FIXED_RANGE

        # floodFill modifies the image, so work on a copy
        tmp = self.imgcontrast.copy()

        num, _, mask_out, rect = cv2.floodFill(
            tmp,
            mask,
            seedPoint=seed,
            newVal=(0, 0, 0),
            loDiff=lo,
            upDiff=up,
            flags=flags
        )

        # Convert OpenCV floodFill mask to binary region mask
        region_mask = ((mask_out[1:-1, 1:-1] != 0).astype(np.uint8) * 255)

        if display:
            #cv2.imshow("Green Mask", self.hole_mask)
            cv2.imshow("FloodFill Region Mask", region_mask)
            #cv2.imshow("FloodFilled Image", tmp)


        print("FloodFill seed:", seed)
        print("Filled pixels:", num)
        print("Bounding rect:", rect)

        return region_mask



# Example
imageproce = ImagePreprocessor("C:\\Users\\marti\\Documents\\DL-Mechanical-Defects\\dataset_creation\\images\\bad\\13_bad_focus_2.jpg")
imageproce.outer_rim_cutting(display=True)
cv2.waitKey(0)
cv2.destroyAllWindows()

# hole_plus_metal = imageproce.image_initial_cutting(display=False)
# imageproce.outer_rim_cutting(display=True)

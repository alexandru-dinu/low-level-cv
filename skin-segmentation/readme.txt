[x] Download 10 images with human faces from the web.

[x] Threshold the image based on a certain hue range, in order to get
regions containing human skin.

[x] Remove the smallest, noisy connected components.

[x] Dilate and erode the remaining blobs.

[ ] Compute regions properties of the remaining blobs (center and
main axes) and keep the sufficiently round ones in order to obtain
human faces. Fit ellipses to those blobs and show them.


- compute hsv histogram for relevant hue range to show particularities for each image
- perform hsv threshholding based on info gathered from histogram
-

best on: 2, 3, 5, 6, 7
[x] Download 10 images with human faces from the web.

[x] Threshold the image based on a certain hue range, in order to get
regions containing human skin.

[ ] Remove the smallest, noisy connected components.

[ ] Dilate and erode the remaining blobs.

[ ] Compute regions properties of the remaining blobs (center and
main axes) and keep the sufficiently round ones in order to obtain
human faces. Fit ellipses to those blobs and show them.
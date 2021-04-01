# Face Detection & Contour Extraction

My realization of face detection & contouring based on Haar-Cascade classifier and combination of simple digital image processing methods.

## Content in Each Folder

### code

- Folder **opencv_xml** includes all the pre-trained xml files provided by cv2 library.
- **main.py**  - aggregated file for the realization of the whole algorithm, contains test_on_camera function.
- **evaluation.py** - makes evaluation of the algorithm, and plots of the results.
- **my_canny_detector.py** - implementation of canny method of contouring.
- **my_fg_bg_detector.py** - implementation of foreground / background  method of contouring.
- **my_haar_cascade.py** - implementation of Haar-Cascade classifier.

### images

- **my_dataset**

  Includes 10 image selected from FEI database.

- **test_imgs**

  Includes test images for the self-use and plots.

- **haar_results**

  Includes Haar-Cascade classifier results on **my_dataset**.

- **contouring_results**

  Includes contouring results from my algorithm on **haar_results**.

- **contouring_compare**

  Includes manually drew contouring on **haar_results** 

- **figures**

  Includes evaluation figures for the report.

- **FEI_front_faces**

  Includes 200 different faces shot (Full body shot) from the FEI database.

### auxiliary_materials

It includes all the auxiliary materials: reference papers and a video.
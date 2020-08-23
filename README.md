# Deep-Blue-Project

Mall Dataset:
https://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html

CNN based density estimation and Crowd counting: https://arxiv.org/pdf/2003.12783.pdf

## Crowd Counting Techniques:

![image.jpg](https://github.com/unnatibshah/Deep-Blue-Project/blob/master/img/Crowd%20Counting%20Techniques.jpg)

#### Counting by Detection

Counting by detection can be defined as a method to compute the abstraction of image information and local decisions at every point to know about features of a particular type at that point. The authors in [33] proposed a CNN-based hybrid hidden Markov model (HHMM) for speech recognition. The HMM is used to obtain inherent dynamic features that can be used for anomaly detection in crowd analysis. The authors in [34,35] found a solution for reconstructing full-body locomotion that could be used in 3D crowd analysis and abnormal-behavior detection. The earlier research focused on detection-based counting to count the number of people in a scene [3]. Through a sliding-window detector, detection could be monolithic or part-based. Traditional pedestrian-detection techniques used monolithic detection [16,36,37,38]. In these techniques, a classifier is trained by using different features, including Histogram Oriented Gradient (HOG) [16], edgelet [39] and a shapelet [40] extracted from the body of people. The monolithic way of detection performs very well in low-density crowds, but its performance degrades in high density. Therefore, researchers were motivated to address this issue by using part-based detection techniques [41,42] that use boosted classifiers for specific body parts, including shoulders and head, to estimate the count in that area [43].

#### Counting by Regression

Counting by regression is carried out to obtain a more robust and accurate function via known inputs of images and output (ground truth). The authors in [34,35] determined a solution on the basis of reconstructing full-body locomotion that could be applicable in 3D crowd analysis and anomaly detection. Regression-based crowd-density estimation was first exploited by Davies et al. [7]. The extraction of low-level features (foreground area and edge features) is carried out in the video frame. The total edge count and foreground area are extracted from the raw features. In this way, a linear-regression model was developed to establish mapping between actual and estimated count. Shape- and part-based detectors are not successful in the presence of high-density crowds and high-clutter backgrounds. The main components that establish counting by a regression pipeline are low-level feature extraction and regression modelling [4]. Different features, such as gradient, foreground, and edge features, and textures are used to encode low-level information. Further, standard background-subtraction techniques are used for the extraction of foreground features that are removed from foreground segments. Blob-based holistic features, such as perimeter, area, and perimeter–area ratio have had promising results [4,25]. However, these techniques focus on the global properties of the scene. Local features and textures like Gray Level Co-Occurence Metrics (GLCM), HOG, and Local Binary Pattern (LBP), are used to further improve the accuracy of classification, detection, and crowd counting. After the extraction of local and global features, a variety of regression methods, including linear [44], Gaussian, [45], and ridge regression [7], and NNs [46] are used to learn mapping between the actual crowd count and low-level features.

#### Counting by Density Estimation

Counting through density estimation is employed to obtain an estimate by using observed data of an unobservable probability-density function. This technique has made it possible to overcome the problem of occlusion and clutter by using spatial information with a density-estimation approach. For example, Lempitsky et al. [10] incorporated spatial information by proposing linear mapping between local features and estimated-density (ED) maps. The difficult task of detecting and localizing individual objects has been eliminated by calculating image density whose integral in any particular region provides the estimated count of that region. In [10], cutting-plane optimization is used to solve convex optimization tasks by introducing a risk-based quadratic cost function.

#### Counting by CNN

Though detection, regression, clustering, and density-estimation-based crowd-counting techniques perform well to some extent by using handcrafted features, for crowd analysis, motion analysis, and the 3D construction of body parts, different types of CNN- and LSTM-based algorithms have been proposed. In particular, the authors in [47] and [48] proposed a CNN-based descriptor and LSTM-based network to obtain motion and appearance information along the tracks of human body parts. Similarly, the authors in [49] investigated 3D face-model construction by using a 2D view of the face. Further, the authors in [50] investigated deep-learning architecture for the classification of a driver’s actions. Abstractive text summary using a generative adversarial network was done by the authors in [51], while the authors in [52] proposed a CNN-based technique to obtain high representational features for the detection of secondary protein structures. In order to further improve accuracy, researchers used CNN-based crowd-counting techniques [21,53,54]. Counting through CNN employs convolution, pooling, Rectified Linear Unit (RelU), and Fully Connected Layers (FCLs) to extract features that are used to obtain the density map [55]. Counting through CNN is more efficient in terms of accuracy, but at the cost of high computational complexity.

#### Counting by Clustering

Counting by clustering relies on the assumption that visual features and individual motion fields are uniform, so similar features are grouped into different categories. For example, [13] used a Kanade–Lucas–Tomasi (KLT) tracker to obtain low-level features, and then employed Bayesian clustering [14] to find the approximate number of people in an image. The aforementioned methods explicitly model appearance features. Thus, false estimation arises when people remain in static position or when objects repeatedly share the same trajectories. Hence, we concluded that counting by clustering performs better in continuous image frames.

## Unique Challenges of CNN-Based Image Crowd Counting

CNN-based crowd counting faces many challenges that restrict the counting accuracy of these networks (i.e., MAE, MSE, and ED) and the resolution of the density map.

![image1.jpg](https://github.com/unnatibshah/Deep-Blue-Project/blob/master/img/challenges.jpg)

* Occlusion occurs when two or more objects come very close to each other and merge, so that it is hard to recognize individual objects. Thus, crowd-counting accuracy is decreased [18].

* Clutter is a kind of nonuniform arrangement of objects that are close to each other. It is also related to image noise, making recognition and counting tasks more challenging [19].

* Irregular object distribution refers to varying density distribution in an image or a video. For irregular objects, counting through detection is only viable in sparse areas. On the other hand, counting by regression overestimates the sparse areas and is only viable in dense areas. Thus, the irregular distribution of an object is a challenging task for crowd counting [20].

* Nonuniform object scales often occur due to different perspectives. In counting, objects close to the camera look larger when compared to ones farther away. The nearest objects have more pixels than far-away objects. Thus, ground-truth and actual-density estimations are affected by the nonuniform pixel distribution of the same object [21].

* An inconstant perspective occurs due to different camera angles, tilt, and the up–down movement of the camera position. Object recognition and counting accuracy are greatly affected by varying perspectives [22].

### Motivation for Employing CNN-Based Image Crowd Counting

Traditional handcrafted crowd-counting techniques such as those in [1,14] perform well if the training dataset has a low computational cost. However, challenges like occlusion, clutter, and scale variation reduce the accuracy of such traditional methods. In addition, the ED map obtained by employing these handcrafted methods has a low resolution that limits their applicability in many areas, such as medical imaging and military applications. In short, the manual nature of feature extraction by handcrafted methods makes them less (non)adaptive to evolving crowd-counting demands. By observing the above-mentioned deficiencies in traditional crowd-counting algorithms, and the success of CNNs in numerous computer-vision applications, researchers were inspired to exploit their ability in estimating the nonlinear feature density maps of crowd images [53,54,55]. These density maps can be utilized in machine-learning processes for more accurate prediction/estimation of the crowd count [63,64]. Further, up- and downsampling, scale aggregation, and preclassification with a multicolumn approach could also be used to increase the accuracy of crowd counting. On the other hand, deconvolution [65] and Generative Adversarial Networks (GANs) [66] can be employed to enhance the quality of a density map for medical applications.

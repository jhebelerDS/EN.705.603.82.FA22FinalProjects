# 705.603 - AI Enabled Systems: Systems Project
Using Sage Continuum's Sage Data Client API for Determing Population Density in the Chicago Loop


<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#instructions">Instructions</a></li>
      </ul>
    </li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

The Systems Project uses Sage Data API Client to query traffic state and object counts in order to train a XGB Classifer to classify the following  5 Wild Nodes: W02C, W07A, W07B, W026, W079. Since the latest registration for these nodes occurred in May 2022, the Sage API client was queried for 05/22 - 12/22. The start time was calculated Time - 5/22/22 at 0:0:0 or 4,912 hours.  

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

To access the specific image that is tied to this repository, please download its associated image. 

<!-- INSTRUCTIONS -->

### Instructions

To download the repository image: 
* copy pull command for latest
  ```sh
  docker pull gonellcl/systems_proj
  ```
* select RUN
* Run a new container for the image. There are no ports exposed in the repository image. 
* The image is now in your host docker repository

* To access the image in your local repository: 
  ```sh
  $ docker run –v <host directory>:/output <REPOSTIORY>:<IMAGENAME>
  ```
   ```sh
  $ docker run –v <host directory>:/output gonellcl/systems_proj:latest
  ```

<!-- CONTACT -->
## Contact

Claribel Gonell  - cgonell1@jh.edu.

Project Link: [https://github.com/gonellcl1/705.603_claribelgonell](https://github.com/gonellcl1/705.603_claribelgonell)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Resources

* [othneildrew](https://github.com/othneildrew/Best-README-Template/blob/master/BLANK_README.md)
* [Object Counter](https://github.com/waggle-sensor/plugin-objectcounter)
* [Traffic State](https://portal.sagecontinuum.org/apps/app/seonghapark/traffic-state)
* [YOLOv7: The Most Powerful Object Detection Algorithm](https://viso.ai/deep-learning/yolov7-guide/)
* [Array of Things](https://arrayofthings.github.io/index.html)
* [Hands-On Computer Science: The Array of Things Experimental Urban Instrument](https://ieeexplore.ieee.org/abstract/document/9734773)
* [Sage Continuum](https://sagecontinuum.org/) 
* [Sage Continuum Portal](https://portal.sagecontinuum.org/nodes) 
* [Waggle@AI](https://docs.waggle-edge.ai/docs/about/overview)
* [Measuring Cities with Software-Defined Sensors](https://ieeexplore.ieee.org/abstract/document/9241512)
* [Waggle: An open sensor platform for edge computing](https://ieeexplore.ieee.org/abstract/document/7808975)
* [Making Sense of Sensor Data: How Local Environmental Conditions Add Value to Social Science Research](https://journals.sagepub.com/doi/abs/10.1177/0894439320920601)
* [Project Eclipse](https://www.microsoft.com/en-us/research/project/project-eclipse/)
* [SAGE: A Software-Defined Sensor Network](https://www.anl.gov/mcs/sage-a-softwaredefined-sensor-network)












<p align="right">(<a href="#readme-top">back to top</a>)</p>


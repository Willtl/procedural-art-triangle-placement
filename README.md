# procedural-art-triangle-placement
Procedural placement of transparent triangles to resemble target painting.

The mean squared error between drawing and target painting is minimized through gradient free optimizers such as Differential Evolution (DE) and Parameter-exploring Policy Gradients (PGPE) with clipped updates (ClipUp).
 
 

### Table 1. Target and painted images with 100 triangles. 
Target |                              DE                              |              DE center init. + PGPE with ClipUp               
:----:|:------------------------------------------------------------:|:-------------------------------------------------------------:
<img src="targets/mona.jpg" width="250" /> |     <img src="results/final/mona_de.png" width="250" />      |     <img src="results/final/mona_pgpe.png" width="250" />     |
<img src="targets/darwin.jpg" width="250" /> |    <img src="results/final/darwin_de.png" width="250" />     |    <img src="results/final/darwin_pgpe.png" width="250" />    |
<img src="targets/lk.jpg" width="250" /> |   <img src="results/final/lk_diff_evo.png" width="250" />    |      <img src="results/final/lk_pgpe.png" width="250" />      |
<img src="targets/lunardi_face.jpg" width="250" /> | <img src="results/final/lunardi_diff_evo.png" width="250" /> | <img src="results/final/lunardi_face_pgpe.png" width="250" /> |
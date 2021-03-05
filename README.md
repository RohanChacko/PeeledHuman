<h1 align="center">PeeledHuman: Robust Shape Representation for Textured 3D Human Body Reconstruction</h1>
<p align="center"><b>International Conference on 3D Vision, 2020</b></p>
<div align="center">
  <span>
    <a href="https://scholar.google.com/citations?user=NtfzxawAAAAJ">Sai Sagar Jinka<sup>1</sup></a>,
    <a href="https://scholar.google.com/citations?user=qYdxs0wAAAAJ">Rohan Chacko<sup>1</sup></a>,
    <a href="https://scholar.google.com/citations?user=4ladtC0AAAAJ">Avinash Sharma<sup>1</sup></a>,
    <a href="https://scholar.google.co.in/citations?user=3HKjt_IAAAAJ">P. J. Narayanan<sup>1</sup></a>
  </span>
</div>
<p align="center"><sup>1</sup>Center for Visual Information Technology, IIIT Hyderabad</p>
<hr>
<img src="https://rohanchacko.github.io/images/motivation.jpg" width="900px" height="319px">
<div align="center">
  <span>
    <a href="https://rohanchacko.github.io/peeledhuman">[Project page]</a>
    <a href="https://rohanchacko.github.io/files/peeledhumans/peeledhuman.pdf">[Paper]</a>
    <a href="https://arxiv.org/abs/2002.06664">[ArXiv]</a>
  </span>
</div>
<hr>
<p><b>Abstract</b><br>
  We introduce PeeledHuman - a novel shape representation of the human body that is robust to self-occlusions. PeeledHuman encodes the human body as a set of Peeled Depth and RGB maps in 2D, obtained by performing raytracing on the 3D body model and extending each ray beyond its first intersection. This formulation allows us to handle self-occlusions efficiently compared to other representations. Given a monocular RGB image, we learn these Peeled maps in an end-to-end generative adversarial fashion using our novel framework - PeelGAN. We train PeelGAN using a 3D Chamfer loss and other 2D losses to generate multiple depth values per-pixel and a corresponding RGB field per-vertex in a dual-branch setup. In our simple non-parametric solution, the generated Peeled Depth maps are back-projected to 3D space to obtain a complete textured 3D shape. The corresponding RGB maps provide vertex-level texture details. We compare our method with current parametric and non-parametric methods in 3D reconstruction and find that we achieve state-of-theart-results. We demonstrate the effectiveness of our representation on publicly available BUFF and MonoPerfCap datasets as well as loose clothing data collected by our calibrated multi-Kinect setup.
</p>

### Testing

Install environment
```
conda env create -f environment.yml
```

Run the inference script
```python
python test.py                            \
  --test_folder_path <path/to/images/dir> \
  --results_dir <path/to/results/dir>     \
  --name <checkpoint name>                \
  --direction AtoB                        \
  --model pix2pix                         \
  --netG resnet_18blocks                  \
  --output_nc 4                           \
  --load_size 512                         \
  --eval
```

The script looks for the checkpoint file in checkpoints/<checkpoint/name>

<p><b>BibTeX</b><br>
  <pre class="bg-light" style="padding: 5px 10.5px;">@inproceedings {jinka2020peeledhuman,
  author = {S. Jinka and R. Chacko and A. Sharma and P. Narayanan},
  booktitle = {2020 International Conference on 3D Vision (3DV)},
  title = {PeeledHuman: Robust Shape Representation for Textured 3D Human Body Reconstruction},
  year = {2020},
  pages = {879-888},
  doi = {10.1109/3DV50981.2020.00098},
  publisher = {IEEE Computer Society},
  }
  </pre>
</p>

### Acknowledgements
Our network derives from the [pix2pix](https://phillipi.github.io/pix2pix/) work and hence builds on the [official PyTorch implementation of pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/). This README template was borrowed from [Aakash Kt](https://github.com/AakashKT). Please open an issue in case of any bugs/queries.

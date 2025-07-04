# glTF Path Tracer
An offline path tracer that leverages CUDA to generate photorealistic renders of glTF models using the GPU. In addition to being massively parallelized, the program also utilizes Bounding Volume Hierarchies and several Importance Sampling strategies (cosine-weighted hemisphere for diffuse and VNDF for specular, whether to sample diffuse or specular is decided by importance sampling fresnel response) to further improve render times. Originally developed 12/2024.

## Usage
This is a command line application. Upon running it you will be prompted to provide several values via stdin, such as the path to the glTF file, an optional environment map (must be equirectangular projection in the `.hdr` format), and various camera parameters. Once all of these parameters have been entered, the program will render the path traced scene and write the result to an output.png file in the working directory.

## Sample Renders
![Sponza](samplerenders/sponza16384.png)
![Unit 01](samplerenders/unit01.png)
![Workshop](samplerenders/workshop65536.png)
![Pavilion](samplerenders/pavilion16384.png)
![Robot](samplerenders/robot65536.png)
![Cornell Box](samplerenders/ogbox.png)
![Cornell Box 2](samplerenders/noroughnessbox.png)
![Cornell Box 3](samplerenders/halfroughnessbox.png)

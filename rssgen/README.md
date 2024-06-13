# A Benchmark Suite for Systematically Evaluating Reasoning Shortcuts

Official repository for the "A Benchmark Suite for Systematically Evaluating Reasoning Shortcuts" paper.

The data generator includes:

- `MNISTMath`
- `MNISTLogic`
- `KandLogic`
- `CLE4EVR`
- `SDDOIA`

> **NOTE:** The `bpy` library is compatible only with `Python 3.7`. If you intend to generate `CLE4EVR` or `SDDOIA` data using this library, ensure that you are using Python version 3.7 for compatibility.

**Remember to run this for bpy**

```
pip install bpy==2.91a0 && bpy_post_install
```

## CLE4EVR and SDDOIA

`CLE4EVR` and `SDDOIA` use Blender:

```
apt-get install blender
```

## Installation and use

Access the linux terminal and use the conda installation followed by pip3:

```
$conda env create -n rs python=3.7
$conda activate rs
$pip install -r requirements.txt
```

## Structure of the Repository

Before running the code you should edit the `conf.yml` file specifying all the characteristic your dataset should have, some examples will be provided in the folder named `examples_conf`.

The repository is structured in the following way:

- `rssgen`: the main module of the repository
- `examples_conf`: contains some examples for the datasets configurations
- `boia_conf`: contains the configuration files for `SDDOIA`
- `clevr_config`: contains the configuration files for `CLE4EVR`
- `rssgen/parsers`: contains the parsers for the yaml content
- `rssgen/generator`: contains the stuff for generating the synthetic datasets

The general flow should be the following:

```
			MAIN
	   --------------
	| ^        | ^			
	v |        v |			   
     parsers   generators
     			

```

## Generate the dataset

First, ensure that you modify the YAML configuration files for KandLogic, MNISTMath, and MNISTLogic. For CLE4EVR and SDDOIA, provide the necessary command-line arguments. The following lines assume these configurations are already set.

### Generate MNISTMath

```
python -m rssgen examples_config/mnist.yml mnist MNIST_MATH_OUT_FOLDER
```

### Generate MNISTLogic

```
python -m rssgen examples_config/xor.yml xor MNIST_LOGIC_OUT_FOLDER
```

### Generate KandLogic

```
python -m rssgen examples_config/kandinsky.yml kandinsky KAND_LOGIC_OUT_FOLDER
```

## Blender data generation

`CLE4EVR` and `SDDOIA` need to be run inside `Blender`. Therefore, please make sure to modify the import lines in `rssgen/clevr/clevr_renderer.py` and `rssgen/sddoia/sddoia.py` to point to the location of the repository on your PC. Additionally, ensure that the import points to the libraries in your environment so that Blender's built-in Python interpreter can access them.

### Generate CLEVR

```
cd rssgen/clevr
Xvfb :1 -screen 0 1024x768x24 & export DISPLAY=:1
blender -noaudio -b -P clevr_renderer.py
```

### Generate SDDOIA

**NOTE:** Due to licensing restrictions, the traffic light models cannot be provided directly in this repository. Therefore, you have two options:

1. **Create Your Own Traffic Light Models**: You can build your own models in Blender and place them in the `boia_config/shapes` folder. Make sure to name the files as follows:
   - `TLG.blend` for the green traffic light,
   - `TLY.blend` for the yellow traffic light, and
   - `TLR.blend` for the red traffic light.
   
   Additionally, the names of the objects inside each file should be `TLG`, `TLY`, and `TLR`, respectively.

2. **Download and Modify Existing Models**: You can download traffic light models from [TurboSquid](https://www.turbosquid.com/3d-models/traffic-light-547022) and modify them to fit the project requirements.

If you need to change the names of the models, you can modify the `config.py` file, which refers to all the models included in the project.

```
cd rssgen/sddoia
Xvfb :1 -screen 0 1024x768x24 & export DISPLAY=:1
blender -noaudio -b -P sddoia.py
```

## Issues report, bug fixes, and pull requests

For all kind of problems do not hesitate to contact me. If you have additional mitigation strategies that you want to include as for others to test, please send me a pull request. 

## Makefile

To see the Makefile functions, simply call the appropriate help command with [GNU/Make](https://www.gnu.org/software/make/)

```bash
make help
```

The `Makefile` provides a simple and convenient way to manage Python virtual environments (see [venv](https://docs.python.org/3/tutorial/venv.html)).

### Environment creation

In order to create the virtual enviroment and install the requirements be sure you have Python 3.8

```bash
make env
source ./venv/reasoning-shortcut/bin/activate
make install
```

Remember to deactivate the virtual enviroment once you have finished dealing with the project

```bash
deactivate
```

### Generate the code documentation

The automatic code documentation is provided [Sphinx v4.5.0](https://www.sphinx-doc.org/en/master/).

In order to have the code documentation available, you need to install the development requirements

```bash
pip install --upgrade pip
pip install -r requirements.dev.txt
```

Since Sphinx commands are quite verbose, I suggest you to employ the following commands using the `Makefile`.

```bash
make doc-layout
make doc
```

The generated documentation will be accessible by opening `docs/build/html/index.html` in your browser, or equivalently by running

```bash
make open-doc
```

However, for the sake of completeness one may want to run the full Sphinx commands listed here.

```bash
sphinx-quickstart docs --sep --no-batchfile --project rssgen --author "X"  -r 0.1  --language en --extensions sphinx.ext.autodoc --extensions sphinx.ext.napoleon --extensions sphinx.ext.viewcode --extensions myst_parser
sphinx-apidoc -P -o docs/source .
cd docs; make html
```

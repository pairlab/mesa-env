# Objects

This repo already contains all objects from LIBERO, all the Objaverse objects from RoboCasa and a subset of the AI-generated ones, which can be found in `mesa/sim/assets/objects`. To add the rest of the RoboCasa AI-generated objects, download them using their download script and move them into the `mesa/sim/assets/objects/robocasa_ai` folder, and they should be immediately available. 

## Visualizing Objects

During development it can be useful to quickly visualize objects. You can do this using the `scripts/visualize.py` script. For example, to visualize an apple from the RoboCasa objects, you can run:

```bash
uv run scripts/visualize.py --object-names robocasa_apple_0
```

If you'd like to interact with the object, perhaps to determine whether it is easily graspable, you can use the `--device` flag to connect a Quest controller or a SpaceMouse:

```bash
uv run scripts/visualize.py --object-names robocasa_apple_0 --device quest
```


# TextBoxes++ Additional Demos
The documents in this repository are based on [TextBoxes_plusplus][tbpp]. In addition to the demos provided by the contributors, I did some extensions.

  - callpy.cpp
    This file call textpy.py using c++, in order to add text detection to other c++ projects.
    It contains a main function for testing. 
    To run this file, use CMakeLists.txt to compile it.
    ```sh
    $ cd <yourDir>/text
    $ cmake .
    $ make
    $./Text
    ```
  - textpy.py
    This is what callpy.cpp calls. It contains both network preparation and text detection.
  - text.py
    This is a modified version of demo.py. 
    In contrast with demo, it performs text detection on a batch of images.

   [tbpp]:<https://github.com/MhLiao/TextBoxes_plusplus.git>

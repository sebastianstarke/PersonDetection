1. Copy the folder in your Hydro Worskapce
2. Type 'cmake .'
3. Type 'make'
4. Launch the package with 'roslaunch person_detection person_detection.launch'
5. Make sure you subscribe to the right camera topic (check it in /launch/person_detection.launch)

- The topic 'persons' publishes true persons as a vector
- The topic 'personMarker' publishes all true and potential persons for visualization in RViz

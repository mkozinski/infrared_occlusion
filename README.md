# how to use this code

Create a new folder
Clone general network training routines and the experiment code into that folder  
`git clone https://github.com/mkozinski/NetworkTraining_py`  
`git clone https://github.com/mkozinski/infrared_occlusion`  

Enter the experiment directory
`cd infrared_occlusion`  

To create dataset split,
run ./split_data.sh <location of the data folder>; (if needed run chmod +x split_data.sh first); 
This generates the files: trainFiles.txt testFiles.txt containing the names of the training and test images

To run training   
`python "run_v1.py"`  
note: you may need to change the batch size in the script (`batch_size`,`num_workers`) if it runs out of memory.
The training progress is logged to a directory called `log_v1`. It can be plotted in gnuplot:  
a) the epoch-averaged loss on the training data `plot "<logdir>/logBasic.txt" u 1`,  
b) the F1 performance on the test data `plot "<logdir>/logF1Test.txt" u 1`.

The training loss and test performance plots are not synchronised as testing is performed once every 100 epochs.

The networks weights are dumped in the log directories:  
a) the recent network is stored at `<logdir>/net_last.pth`,  
b) the network attaining the highest F1 score on the test set is stored at `<logdir>/net_Test_bestF1.pth`.

To generate prediction for the test set using a trained network run  
`python "segmentTestSet.py"`.  
The name of the file containing the network, and the output directory are defined at the beginning of the script.
a) The output is saved in the form of png images in `log_v1/output_best_v1/`.
b) The precision-recall plot is saved in `log_v1/output_best_v1_pr.txt`.

To create the precision-recall plot, run
`gnuplot plot_precision_recall.gp`



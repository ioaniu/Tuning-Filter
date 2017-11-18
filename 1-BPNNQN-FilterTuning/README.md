
# BPNNQN-FilterTuning

A Intelligent Method of Tuning Filter Based on Q-Learning

### Installation Dependencies:
* Python 2.7
* Keras 1.0

### Directories included in the toolbox

`ai/`   - Back Propagation Neural Network with Keras.

`cavityfilter/`  - The environment of tuning filter by simulating. 

`data/`  - Data used by simulating the environment of tuning filter.

`memory/`  - Utility functions used to sorce parameters.

`log/` - Save the log files in which processes are recorded.

`documents/` - Published papers.


### Algorithm 

>
> Initialize replay memory *D* to capacity N  
> Initialize action-value function *Q* with random weights *w* and biases *b*  
>  Initialize target action-value function <img src="https://latex.codecogs.com/gif.latex?Q_{target}" title="Q_{target}" /> with weights <img src="https://latex.codecogs.com/gif.latex?w_{t}&space;=&space;w" title="w_{t} = w" />  
>  **For** *episode* = 1, *M* **do**  
>  &emsp;&emsp;Initialize sequence <img src="https://latex.codecogs.com/gif.latex?s_{t}" title="s_{t}" /> (S-parameters)  
>  &emsp;&emsp;**For** *t* = 1, *T* **do**  
>  &emsp;&emsp;&emsp;&emsp;With probability *e* select *a* random action <img src="https://latex.codecogs.com/gif.latex?a_{t}" title="a_{t}" />  
>  &emsp;&emsp;&emsp;&emsp;Execute action <img src="https://latex.codecogs.com/gif.latex?a_{t}" title="a_{t}" /> in filter tuning and observe reward <img src="https://latex.codecogs.com/gif.latex?r_{t}" title="r_{t}" /> and <img src="https://latex.codecogs.com/gif.latex?s_{t&plus;1}" title="s_{t+1}" />  
>  &emsp;&emsp;&emsp;&emsp;Store transition <img src="https://latex.codecogs.com/gif.latex?\left&space;(&space;s_{t},&space;a_{t},&space;r_{t},&space;s_{t&plus;1}&space;\right&space;)" title="\left ( s_{t}, a_{t}, r_{t}, s_{t+1} \right )" /> in *D*  
>  &emsp;&emsp;&emsp;&emsp;Sample random minibatch of transitions from *D*  
>  &emsp;&emsp;&emsp;&emsp;Compute <img src="https://latex.codecogs.com/gif.latex?y_{t}&space;=&space;r_{t}&space;&plus;&space;max_{a_{t&plus;1}}Q^{'}\left&space;(&space;s_{t&plus;1},&space;a_{t&plus;1};&space;w,b&space;\right&space;)" title="y_{t} = r_{t} + max_{a_{t+1}}Q^{'}\left ( s_{t+1}, a_{t+1}; w,b \right )" />  
>  &emsp;&emsp;&emsp;&emsp;Perform a gradient descent to update *w*, *b*  
>  &emsp;**end for**  
>  **end for**


### How to Run?

> git clone https://github.com/ioaniu/Tuning-Filter.git  
> cd 1-BPNNQN-FilterTuning  
> python main.py


### Other code
'matlab/'   - Matlab Code.

---


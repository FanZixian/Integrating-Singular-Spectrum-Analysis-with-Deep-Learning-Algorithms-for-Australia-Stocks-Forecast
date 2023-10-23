# Integrating Singular Spectrum Analysis with Deep Learning Algorithms for Australian Stock Forecast

The project aims to implement SSA method to the stocks and provide a trading strategy with the support of deep learning algorithms.

## Description

This paper proposes a model combining Singular Spectrum Analysis (SA) and Deep Learning (DL), targeting stocks on ASX 50 in the Australian market. First, stock prices are decomposed into valuable sequences and eliminating noise, and then multiple DLs are trained using the denoised series to make forecasts and construct appropriate trading strategies. The experimental results show that the SSA-DL models can uncover valid information in stock prices, construct denoised stock price prediction data, and obtain better prediction results as well as investment returns. The best model obtained in this study was SSA-CNN-LSTM, which was able to generate a Sharpe Ratio of up to 1.88 and 67% ROI.

## Getting Started

### Dataset description (Data repository)

* All the original data of stocks price are extracted from [yahoo finance](https://finance.yahoo.com/), dates from 1/2/2018 until 3/31/2023. And the data are stored in ASX_stockconsidering repository.
* The Grouped repository contains data after grouping according to sectors, and also data eliminated based on start time.
* Generated repository contains dataset after processing from SSA algorithm.
* Prediction repository only contains prediction result from CNN model.
* Parameter tuning repositories contain results from all the other models with different hyper parameters.
* MAE_MSE repository contains evaluation metrics.

# Models' Code
* Files to gerenate available stock data
* Files to implement SSA algorithm
* Files to implement different deep learning algorithms
* Files to evaluate the models' performance

### Trading Strategy Repository
* After obtaining all the results from the models' prediction, these codes are run to provide trading startegy results and financial metrics for further domain evaluation.

### Installing

```
pip install -r requirements.txt
```

### Executing program

Initially, please refer to the rar file contained in the repository, download and unrar it. It is because there are multiple empty directories contained in the research, which is to let the user create the train and test files by running the code directly, instead of providing the files in advance (as it is required by the reviewer to show that the train-test split are not influenced by the FUTURE INFORMATION PROBLEM). And github won't allow to push empty directories unless it contains the .gitkeep file.

To download all the files in the repo, you can simply use git clone:

```
git clone https://github.com/FanZixian/Integrating-Singular-Spectrum-Analysis-with-Deep-Learning-Algorithms-for-Australia-Stocks-Forecast.git
```

Simply open every ipynb file and run them with the right kernel.

The formal sequence of running the program is:
1. Make sure you have all the repositories as mentioned in the `data_directory_structure.txt` file, otherwise some of the code cannot find the existing paths.
2. Run the `Stock Generator.ipynb` to generate the stocks and seperate them into different groups.
3. Run the `Tune_Set_Generation_9995.ipynb` and `Tune_Set_Generation_9995.ipynb` to generate the train and test sets of the stocks with the application of SSA algorithm to them. Note that the `print` and `plot`  functions are commented out to prevent it from providing too many messages that can potentially break the jupyter notebook.
4. 



## Authors

Contributors names and contact info

[Fan Zixian](http://linkedin.com/in/zixian-demitry-fan-607611212): u3577161@connect.hku.hk

## License

The data is released under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/), and the code is available under the [MIT License](https://opensource.org/license/mit/). - see the `LICENSE.md` files for details
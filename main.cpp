#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>

#define InNode 2
#define HideNode 4
#define outNode 1

double rate = 0.8 ;
double threshold = 1e-4;
size_t mosttimes = 1e6;

// 定义输入输出样本
struct Sample
{
    std::vector<double> in ,out;
};
// 定义单个神经元
struct Node
{
    double value{} ,bias{} ,bias_delta{};
    std::vector<double> weight ,weight_delta;
};
// 定义工具类
namespace utils
{
    inline double sigmoid(double x)
    {
        double res = 1.0 /(1.0 + std::exp(-x));
        return res;
    }

    std::vector<double> getFielData(std::string fielname)
    {
         std::vector<double> res;
         //创建文件流
         std::ifstream in(fielname);
         if (in.is_open())
         {
            while (!in.eof())
            {
                double buffer ;
                in >> buffer ;
                res.push_back(buffer);
            }
            in.close();
         }
        else
         {
             std::cout<< " Error in reading " << fielname <<std::endl;
         }
         return res;
    }

    std::vector<Sample> getTrainData(std::string fielname)
    {
        std::vector<Sample> res;

        //创建临时变量存放训练数据
        std::vector<double> buffer = getFielData(fielname);

        for (size_t i = 0; i <buffer.size(); i +=InNode + outNode)
        {
     
            // sample中含有in ,out
            Sample tmp;
            for (size_t j = 0; j < InNode; j++)
            {
          //读取in数据
                tmp.in.push_back(buffer[i + j]);
            }

            for (size_t j = 0; j < outNode; j++)
            {
          // 读取out 数据
                tmp.out.push_back(buffer[i + InNode +j]);
            }
            
            res.push_back(tmp);
        }
        
        return res;
    }

    std::vector<Sample> getTestData(std::string fielname)
    {
        std::vector<Sample> res;
        //创建临时变量存放测试数据
        std::vector<double> buffer = getFielData(fielname);
        for (size_t i = 0; i < buffer.size(); i+= InNode)
        {
            Sample tmp;
            for (size_t j = 0; j < InNode; j++)
            {
         
                tmp.in.push_back(buffer[i+j]);
            }
            res.push_back(tmp);
        }
        
        return res;
    }
}

Node *inputlayer[InNode], *hideLayer[HideNode],*outLayer[outNode];

// 初始化权值和偏置值
inline void init()
{
    // 生成随机数
    std::mt19937 rd;
    rd.seed(std::random_device()());

    std::uniform_real_distribution<double> distribution(-1.0,1.0);

    // 初始化输入到隐藏层的权值
    for (size_t i = 0; i < InNode; i++)
    {
        ::inputlayer[i] = new Node();
        for (size_t j = 0; j < HideNode; j++)
        {
            ::inputlayer[i]->weight.push_back(distribution(rd));
            ::inputlayer[i]->weight_delta.push_back(0.f);
        }    
    }
    // 初始化隐藏层到输出层的偏置值和权值
    for (size_t i = 0; i < HideNode; i++)
    {
        ::hideLayer[i] = new Node();
        ::hideLayer[i]->bias = distribution(rd);
        for (size_t j = 0; j < outNode; j++)
        {
            ::hideLayer[i]->weight.push_back(distribution(rd));
            ::hideLayer[i]->weight_delta.push_back(0.f);
        }   
    } 
    // 初始化输出层的偏置值
    for (size_t i = 0; i < outNode; i++)
    {
        ::outLayer[i] = new Node();
        ::outLayer[i]->bias = distribution(rd);
    }
    
}

// 重置权值和偏置值的delta值
inline void reset_delta()
{
    for (size_t i = 0; i < InNode; i++)
    {
 
        ::inputlayer[i]->weight_delta.assign(::inputlayer[i]->weight_delta.size(),0.f);
    }

    for (size_t i = 0; i < HideNode; i++)
    {
        ::hideLayer[i]->bias_delta = 0.f;
        ::hideLayer[i]->weight_delta.assign(::hideLayer[i]->weight_delta.size(),0.f);
    }

    for (size_t i = 0; i < outNode; i++)
    {
        ::outLayer[i]->bias_delta = 0.f;
    }  
    
}


int main (int argc, char *argv[])
{
    init();

    // 读取训练数据
    std::vector<Sample> train_data = utils::getTrainData("traindata.txt");
    //开始迭代
    for (size_t times = 0; times < mosttimes; times++)
    {
        reset_delta();

        double error_max = 0.f;

        for (auto &idx : train_data)
        {
            // 拿到输入层的值
            for (size_t i = 0; i < InNode; i++)
            {
                ::inputlayer[i]->value = idx.in[i];
            }

            // 正向传播
            // 输入层到隐藏层的正向传播
            for (size_t j = 0; j < HideNode; j++)
            {
                double sum = 0.f;
                for (size_t i = 0; i < InNode; i++)
                {
                    sum += ::inputlayer[i]->value * ::inputlayer[i]->weight[j];
                }
                sum -= ::hideLayer[j]->bias;

                ::hideLayer[j]->value = utils::sigmoid(sum);
            }
            // 隐藏层到输出层的正向传播
            for (size_t j = 0; j < outNode; j++)
            {
                double sum = 0;
                for (size_t i = 0; i < HideNode; i++)
                {
                    sum += ::hideLayer[i]->value * ::hideLayer[i]->weight[j];
                }
                sum -= ::outLayer[j]->bias;

                ::outLayer[j]->value =  utils::sigmoid(sum);
            }
            // 计算误差
            double error = 0.f;
            for (size_t i = 0; i < outNode; i++)
            {
                double tmp = std::fabs(idx.out[i] - ::outLayer[i]->value);
                error += tmp * tmp / 2;
            }

            error_max = std::max(error, error_max);

            // 反向传播
            // 输出层的偏置值delta修正
            for (size_t i = 0; i < outNode; i++)
            {
                double bias_delta = -(idx.out[i]-::outLayer[i]->value)* 
                ::outLayer[i]->value * (1.0 - ::outLayer[i]->value);

                ::outLayer[i]->bias_delta += bias_delta;
            }
            // 隐藏层到输出层的权值delta修正
            for (size_t i = 0; i < HideNode; i++)
            {
                for (size_t j = 0; j < outNode; j++)
                {
                    double weight_delta = (idx.out[j]- ::outLayer[j]->value) *
                    ::outLayer[j]->value * (1.0 - ::outLayer[j]->value) *
                    ::hideLayer[i]->value;

                    ::hideLayer[i]->weight_delta[j] += weight_delta;
                }
                
            }
            // 隐藏层到输出层偏置值delta的修正
            for (size_t i = 0; i < HideNode; i++)
            {
                double sum = 0.f;
                for (size_t j = 0; j < outNode; j++)
                {
                    sum += -(idx.out[j] - ::outLayer[j]->value) * 
                    ::outLayer[j]->value * (1.0- ::outLayer[j]->value) *
                    ::hideLayer[i]->weight[j] ;
                }
                ::hideLayer[i]->bias_delta += sum * 
                ::hideLayer[i]->value * (1.0 - ::hideLayer[i]->value);
            }
            // 输入层到隐藏层权值delta的修正
            for (size_t i = 0; i < InNode; i++)
            {
                for (size_t j = 0; j < HideNode; j++)
                {
                    double sum = 0.f;
                    for (size_t k = 0; k < outNode; k++)
                    {
                 
                        sum += (idx.out[k]- ::outLayer[k]->value) *
                        ::outLayer[k]->value * (1.0 - ::outLayer[k]->value) *
                        ::hideLayer[j]->weight[k];
                    }
                    
                    ::inputlayer[i]->weight_delta[j] += 
                    sum * 
                    ::hideLayer[j]->value * (1.0 -::hideLayer[j]->value) * 
                    ::inputlayer[i]->value;
                }
                
            }
            
               
        }
        
        if (error_max < ::threshold)
        {
            std::cout<< "Success with " << times + 1 <<" times traing " <<std::endl;
            std::cout<< "error_max = " << error_max << std::endl;
            break;
        }
        
        // 调整权值和偏置值
        auto train_data_size = double(train_data.size());
        // 调整输出层到隐藏层的权值
        for (size_t i = 0; i < InNode; i++)
        {
            for (size_t j = 0; j < HideNode; j++)
            {
                ::inputlayer[i]->weight[j] += rate * 
                ::inputlayer[i]->weight_delta[j] /train_data_size;
            }
        }
        // 调整隐藏层的权值和隐藏层到输出层的权值
        for (size_t i = 0; i < HideNode; i++)
        {
            ::hideLayer[i]->bias += rate * 
            ::hideLayer[i]->bias_delta / train_data_size;
            for (size_t j = 0; j < outNode; j++)
            {
                ::hideLayer[i]->weight[j] += rate *
                ::hideLayer[i]->weight_delta[j] / train_data_size;
            }
            
        }
        // 调整输出层的偏置值
        for (size_t i = 0; i < outNode; i++)
        {
            ::outLayer[i]->bias += rate * 
            ::outLayer[i]->bias_delta / train_data_size;
        }
    }
    //预测
    // 读取参数 
    std::vector<Sample> test_deta = utils::getTestData("testdata.txt");

    for (auto & idx : test_deta)
    {
        /* code */
        // 读取参数
        for (size_t i = 0; i < InNode; i++)
        {
            /* code */
            ::inputlayer[i]->value = idx.in[i];
        }
        // 开始计算参数
        // 从输出层到隐藏层的参数计算
        for (size_t j = 0; j < HideNode; j++)
        {
            /* code */
            double sum = 0.f;
            for (size_t i = 0; i < InNode; i++)
            {
                /* code */
                sum += ::inputlayer[i]->value * ::inputlayer[i]->weight[j];
            }
            sum -= ::hideLayer[j]->bias;
            
            ::hideLayer[j]->value = utils::sigmoid(sum);
        }
        // 从隐藏层到输出层的参数计算
        for (size_t j = 0; j < outNode; j++)
        {
            /* code */
            double sum = 0.f;
            for (size_t i = 0; i < HideNode; i++)
            {
                /* code */
                sum +=::hideLayer[i]->value * ::hideLayer[i]->weight[j];
            }
            sum -= ::outLayer[j]->bias;

            ::outLayer[j]->value = utils::sigmoid(sum);
            idx.out.push_back(::outLayer[j]->value);

            for(auto &tmp :idx.in)
            {
                std::cout << tmp <<" ";
            }
            for (auto &tmp :idx.out)
            {
                std::cout<< tmp << " " ;
            }
            std::cout << std::endl;
        }
    }
    
    return 0;
    
}
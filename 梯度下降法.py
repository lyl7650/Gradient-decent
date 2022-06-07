# coding : utf-8

# 瞎写梯度下降法

def func(x):
    return x**2 + 2*x

def d_func(x):
    return 2*x + 2


def Gradient_desc(current_x,lr,e,step):
    step = 0
    while d_func(current_x) > e: # 梯度小于e时终止循环
        d = d_func(current_x) # 方向   
        new_x = current_x - lr*d
        min_func = func(new_x)
        print("第{}步的点为{}，值为{}".format(step+1,new_x,min_func))
        current_x = new_x
        step+=1


if __name__ == "__main__":
    current_x = 0
    Gradient_desc(current_x, lr= 0.02, e=0.01,step=50)


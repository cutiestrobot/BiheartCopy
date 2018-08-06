# BiheartCopy
> ### **2018.07.31** 
* 服务器上的涉及到pytorch版本的所有bug 都来源于没法翻墙  
* 慎用清华镜像源 用之前检查一下更新到的版本号  据观察tuna上的 pytorch 只更新到去年七月  
* conda 的channel可换 在channel设置文件里把defaults去掉 就只会从文件里中写好的url中选择包下载  
* 已完成： 建好了conda 虚拟环境 bimindconda  有个坑要注意 在新建环境时候  有个选项叫 no-site-packages  加了这个语句建立的虚拟环境是完全干净不会调用root（及系统默认的site-packages的） 没加的话会默认调用root环境的所有包 在这种条件下 import torch 调用的是root环境的pytorch 包（0.3）
</br>解决方法（我觉得很糟糕） 进入python后 import sys 查看sys.path 用sys.path.remove('/somepath') 去掉默认环境的包的路径
</br>pytorch0.4 已经通过conda install --offline xxxx.tar.bz2 安装成功</br>
* 问题待解决： import torch 时 显示导入的numpy包发生错误 
* 今天尝试了卸载numpy pytorch torchvision  没有解决
* plan for tomorrow： ~~顺利回家~~ （mai tu te chan） o(*￣▽￣*)ブ  
  > 其实是 查一下torch 依赖的是哪个numpy 版本 从报错上来看 应该不是最新版


> ### **2018.08.02**
* plan for today : figure out why the numpy error and fix it

</br>

> ### **2018.08.04**
* 这是清华镜像凉了 连python3.6都装不了的一天
</br>

> ### **2018.08.05**
* 解决了服务器上pytorch0.4 的安装问题
* 服务器连不上网没关系 ， clone 一下root 环境 ， 再把torch重新安装解决问题


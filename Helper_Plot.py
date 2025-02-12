import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')
import matplotlib.colors as mcolors
from matplotlib.font_manager import FontProperties
# import seaborn as sns
import pandas as pd
plt.rcParams['font.family'] = 'Times New Roman'
def single_model_draw_test_CY25_1_plt(real_data,pred_data_1,pred_data_2,pred_data_3,save_filename,save_figure_dir,Rated_Capacity,model):     
    otf_font_path_1 = 'fonts/otf/Times-Newer-Roman-Regular.otf' # 替换为您的OTF字体文件的实际路径          
    font_prop_1a = FontProperties(fname=otf_font_path_1,size=12)      
    font_prop_1b = FontProperties(fname=otf_font_path_1,size=15)    
    otf_font_path_2 = 'fonts/otf/Times-Newer-Roman-Bold.otf'   
    font_prop_2 = FontProperties(fname=otf_font_path_2,size=16)
    fig, ax = plt.subplots()
    x = [t+1 for t in range(len(real_data))]
    threshold = [Rated_Capacity*0.7] * len(real_data)
    # print(plt.rcParams['font.size'])
    ax.set_xlim([0,1000])
    ax.set_ylim([0.8,2.6])
    plt.xticks(fontproperties=font_prop_1a)
    plt.yticks(fontproperties=font_prop_1a)
    ax.set_xticks([0,200,400,600,800,1000])
    ax.set_yticks([0.8,1.1,1.4,1.7,2.0,2.3,2.6])
    ax.plot(x,real_data,color='b',label='Real data',linewidth=1,zorder=2)#marker='>',markersize=1,markevery=(1, 5),
    ax.plot(x[200-1:],pred_data_1,color=mcolors.CSS4_COLORS['red'],label="SP = 200",linewidth=1,zorder=2.5)#marker='s',markersize=1,markevery=(1, 5),
    ax.plot(x[300-1:],pred_data_2,color=mcolors.CSS4_COLORS['yellow'],label="SP = 300",linewidth=1,zorder=3)#marker='.',markersize=1,markevery=(1, 5),
    ax.plot(x[400-1:],pred_data_3,color=mcolors.CSS4_COLORS['lime'],label="SP = 400",linewidth=1,zorder=3.5)#marker='v',markersize=1,markevery=(1, 5),
    ax.plot(x, threshold, mcolors.CSS4_COLORS['black'], linestyle='--',linewidth=1,zorder=4)
    
    axins = ax.inset_axes([0.1, 0.1, 0.47, 0.47])
    # 在子图中绘制相同的曲线
    axins.plot(x,real_data,color='b',marker='>',label='Real data',linewidth=1,markersize=2,zorder=2)
    axins.plot(x[200-1:],pred_data_1,color=mcolors.CSS4_COLORS['red'],marker='s',label="SP = 200",linewidth=1,markersize=2,zorder=2.5)
    axins.plot(x[300-1:],pred_data_2,color=mcolors.CSS4_COLORS['yellow'],marker='.',label="SP = 300",linewidth=1,markersize=2,zorder=3)
    axins.plot(x[400-1:],pred_data_3,color=mcolors.CSS4_COLORS['lime'],marker='v',label="SP = 400",linewidth=1,markersize=2,zorder=3.5)
    axins.plot(x, threshold, mcolors.CSS4_COLORS['black'], linestyle='--',linewidth=1,zorder=4)
    # 调整子图的坐标轴以聚焦特定区域
    axins.set_xlim(765, 795)
    axins.set_ylim(1.73, 1.77)
    plt.xticks(fontproperties=font_prop_1a)
    plt.yticks(fontproperties=font_prop_1a)
    axins.set_xticks([765,775,785,795])
    axins.set_yticks([1.73,1.75,1.77])
    # 添加连接线，指示主图和子图之间的关系
    ax.indicate_inset_zoom(axins, edgecolor="black")
    ax.text(500,2.5,model,horizontalalignment='center',verticalalignment='center', color='k',fontsize=16,fontproperties=font_prop_2)
    # plt.title('real v.s. prediction of battery ' + args.test_name, fontproperties=font_prop_1b)
    ax.set_xlabel('Cycle',fontsize=15, fontproperties=font_prop_1b)
    ax.set_ylabel('Capacity (Ah)',fontsize=15, fontproperties=font_prop_1b)
    lines = [ax.get_lines()[0], ax.get_lines()[1], ax.get_lines()[2], ax.get_lines()[3]]
    ax.legend(lines,['Real data', "SP = 200", "SP = 300", "SP = 400"],loc='upper right', frameon=True, prop=font_prop_1a) # 添加图例,并移除边框
    # fig.savefig(os.path.join(save_figure_dir,save_filename+'.jpg'),dpi=1000)
    fig.savefig(os.path.join(save_figure_dir,save_filename+'.png'),dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(save_figure_dir,save_filename+'.svg'), bbox_inches='tight')#SVG文件的内容本身并不包含分辨率信息
 

if __name__ == '__main__':
      import argparse
      import torch
      parser = argparse.ArgumentParser()
      parser.add_argument('--model', default='iTransformer',help='Model name.')
      parser.add_argument('--seq_len', type=int, default=64, help='input sequence length')
      parser.add_argument('--label_len', type=int, default=0, help='start token length')
      parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length') 
      parser.add_argument('--Battery_list', type=list, default=['CY25_1', 'CY25_2', 'CY25_3'], help='Battery data')
      parser.add_argument('--data_dir', type=str, default='data/TJU data/Dataset_3_NCM_NCA_battery/', help='path of the data file')
      parser.add_argument('--Rated_Capacity', type=float, default=2.5, help='Rate Capacity')
      parser.add_argument('--test_name', type=str, default='CY25_3', help='Battery data used for test')
      parser.add_argument('--start_point_list', type=int, default=[200,400,600], help='The cycle when prediction gets started.')
      parser.add_argument('--seed', type=int, default=1, help='Random seed.')
      parser.add_argument('--root_dir', type=str, default='TJU_RUL_prediction_sl_64', help='root path of the store file')
      parser.add_argument('--count', type=int, default=1, help='The number of independent experiment.')
      parser.add_argument('--batch_size', type=int, default=128, help='The batch size.')
      parser.add_argument('--max_epochs', type=int, default=200, help='max train epochs')

      parser.add_argument('--patch_len', type=int, default=2, help='patch length')
      parser.add_argument('--d_model', type=int, default=16, help='hidden dimensions of model')           

      args = parser.parse_args()

      real_data = torch.load('results/Capacity_CY25_3_real_data.pth')
      predict_results = torch.load('results/RUL_CY25_3_iTransformer.pth')
      root_dir = 'results_{}/{}/{}/'.format(args.root_dir,args.test_name,args.model)
      if not os.path.exists(root_dir):
            os.makedirs(root_dir)
      save_figure_dir = os.path.join(root_dir,'figures')
      if not os.path.exists(save_figure_dir):
            os.makedirs(save_figure_dir)
      if args.test_name == 'CY25_3':
            for i in range(args.count):
                  single_model_draw_test_CY25_1_plt(real_data,predict_results['SP200'][i],predict_results['SP400'][i],predict_results['SP600'][i],
                        save_filename='best_model_{}'.format(i)+'_RUL_Prediction',save_figure_dir=save_figure_dir,Rated_Capacity=args.Rated_Capacity,model=args.model)
    
    
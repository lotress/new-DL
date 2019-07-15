option = dict(edim=16, epochs=10, maxgrad=1., sdt=1e-2, sdt_decay_step=3, batchsize=32)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
zhfont= mpl.font_manager.FontProperties(fname=fontPath)
columns = 2

def drawAttention(indices, l, _, att, *args):
  if len(att.shape) != 3:
    return
  heads = att.size(0)
  l = int(l)
  rows = (heads + columns - 1) // columns
  indices = indices[:l].tolist()
  ticks = np.arange(0, l)
  labels = [''] + [vocab[i] for i in indices]
  fig = plt.figure(figsize=(16, rows * 16 // columns))
  for t in range(heads):
    ax = fig.add_subplot(rows, columns, t + 1)
    data = att[t, :l, :l+1].detach().to(torch.float).cpu().numpy()
    cax = ax.matshow(data, interpolation='nearest', cmap='hot', vmin=0, vmax=1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_xticklabels(labels + ['NA'], fontproperties=zhfont)
    ax.set_yticklabels(labels, fontproperties=zhfont)
  return plt.show()

option['drawVars'] = drawAttention

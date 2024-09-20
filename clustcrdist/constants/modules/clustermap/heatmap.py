#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.cluster import hierarchy
from collections import ChainMap
from .exception import ClusterMapError

class ClusterMap:
    
    def __init__(
            self, 
            data,
            labels = None,
            method = "ward", 
            metric = "euclidean",
            col_dendrogram = True,
            row_dendrogram = False,
            title = None,
            xdim = 3,
            ydim = 5
            ):
        
        # Data
        if isinstance(data, np.ndarray):
            self.labels = labels
            self.matrix = data
        else:
            self.matrix = data.to_numpy()
            self.labels = data.columns
        self.title = title
        np.fill_diagonal(self.matrix, 0)
        
        # Clustering params
        self.method = method
        self.metric = metric
        self.cd = col_dendrogram
        self.rd = row_dendrogram
        
        # Figure setup
        self.fig = plt.figure(
            figsize = (xdim,ydim), 
            dpi = 200
            )
        
        self.gs0 = mpl.gridspec.GridSpec(
            nrows = 2,
            ncols = 1, 
            figure = self.fig,
            height_ratios = [8,2],
            hspace = 0.05
            )
        
        self.gs1 = mpl.gridspec.GridSpecFromSubplotSpec(
            nrows = 2, 
            ncols = 1, 
            subplot_spec = self.gs0[0],
            height_ratios = [2,8],
            hspace=0
            )
        
        self.ax_heatmap = self.fig.add_subplot(self.gs1[1])
        
        self.ax_col_dendrogram = self.fig.add_subplot(
            self.gs1[0], 
            sharex = self.ax_heatmap
            )
        
        self._makeplot()
        
        plt.close()
    
    def add_metadata(self, meta, columns, custom_id_map = None):
        
        self.meta = meta
        self.annot_lab = columns
        
        try: 
            self.meta["meta"] = self.meta["sample"].astype("category")
        except KeyError:
            raise ClusterMapError("Meta data must contain sample column.")
            
        self.meta["sample"] = self.meta["sample"].astype("category")
        self.meta["sample"] = self.meta["sample"].cat.set_categories(self.labels)
        self.meta = self.meta.sort_values(["sample"]).reset_index(drop=True)
        
        if custom_id_map is not None:
            for col in columns:
                self.meta[col+"_id"] = self.meta[col].map(custom_id_map[col])
        else:
            for col in columns:
                self.meta[col+"_id"] = self.meta[col].map(
                    dict(
                        ChainMap(
                            *[{i:n} for n,i in enumerate(
                                self.meta[col].unique()
                                )]
                            )
                        )
                    )
            
                
        self.gs2 = mpl.gridspec.GridSpecFromSubplotSpec(
            nrows = len(columns), 
            ncols = 1, 
            subplot_spec = self.gs0[1]
            )
            
    def _clustering(self):
        return hierarchy.linkage(
            self.matrix, 
            method = self.method, 
            metric = self.metric
            )
    
    def _add_dendrogram(self, links):
        
        self.col_dendrogram = hierarchy.dendrogram(
            links, 
            ax = self.ax_col_dendrogram,
            color_threshold = 0,
            above_threshold_color="black"
            )
        self.row_dendrogram = hierarchy.dendrogram(
            links, 
            no_plot = True
            )
        
        self.ax_col_dendrogram.set_axis_off()

        self.xind = self.col_dendrogram['leaves']
        self.yind = self.row_dendrogram['leaves']
        
        self.xmin, self.xmax = self.ax_col_dendrogram.get_xlim()
        
        self.ax_col_dendrogram.set_title(self.title)
    
    def _heatmap(self):
        
        # Sort matrix according to clustering
        rearranged = pd.DataFrame(self.matrix).iloc[self.xind,
                                                    self.yind].T
        
        self.im = self.ax_heatmap.imshow(
            X = rearranged, 
            aspect = 'auto', 
            extent = [self.xmin,self.xmax,0,1], 
            cmap = 'Spectral_r'
            )
        
        # self.ax_heatmap.yaxis.tick_right()
        
        if self.labels is not None:
            sorted_labels = [self.labels[i] for i in self.xind]
            l = len(self.labels)
            print(l)
            self.ax_heatmap.set_xticklabels(sorted_labels, fontsize=3, rotation=90)
            yticks = self.ax_heatmap.get_yticks()
            # self.ax_heatmap.set_yticks(np.arange((1/l)/2,1.01-(1/l)/2,1/l))
            self.ax_heatmap.set_yticks(np.arange(0,1,1/l))
            self.ax_heatmap.set_yticklabels(labels=sorted_labels[::-1], fontsize=3)
        
            # self.ax_heatmap.set_xticklabels(labels = lab, rotation = 90)
            # plt.setp(
            #     self.ax_heatmap.get_yticklabels(),
            #     rotation = 0,
            #     fontsize = 5
            #     )

            plt.setp(self.ax_heatmap.get_xticklabels(), visible=True)
        # self.fig.show()
        
    def _add_colorbar(self):
        self.cax = self.fig.add_axes(
            [.915, 
             .3, 
             0.025, 
             0.45]
            )
        self.fig.colorbar(self.im, cax=self.cax)
        self.cax.tick_params(labelsize=7) 
        # self.cbarticks = self.cax.get_yticklabels()
        # self.cax.set_yticklabels(
        #     self.cbarticks,
        #     size = 10
        #     )
        
    def _makeplot(self):
        
        links = self._clustering()
        self._add_dendrogram(links)
        self._heatmap()
        self._add_colorbar()
        
    def annotateplot(self, colormaps = None, custom_labs = None):
        
        if colormaps == None:
            colormaps = ["Set1"] * len(self.annot_lab) 

        plt.setp(self.ax_heatmap.get_xticklabels(), visible=True)
            
        for i, (lab,clr) in enumerate(zip(self.annot_lab, colormaps)):
            
            data = np.vstack(
                [self.meta[lab+"_id"][self.xind],
                 self.meta[lab+"_id"][self.xind]]
                )
            values = np.unique(data.ravel())
            print(values)
            
            ax = self.fig.add_subplot(
                self.gs2[i], 
                sharex = self.ax_heatmap)
            
            print(clr)
            print(data)
            im = ax.imshow(
                data, 
                aspect = 'auto', 
                extent = [self.xmin,self.xmax,0,1], 
                cmap = clr
                )
            
            colors = [im.cmap(im.norm(value)) for value in values]
            print(colors)
            patches = [mpl.patches.Patch(
                color = colors[i], 
                label = "{}".format(
                    self.meta[self.meta[lab+"_id"]==i][lab].unique()[0]
                    )
                ) for i in range(len(values))]

            
            
            if custom_labs is not None:
                lab = custom_labs[i]
            else:
                pass
            
            ax.legend(
                handles = patches, 
                # bbox_to_anchor = (1.2, 10 - i), 
                loc = 2, 
                borderaxespad = 0.5,
                fontsize = 6,
                title = lab)
            
            ax.set_yticks([])
            ax.set_ylabel(
                lab, 
                rotation = 0, 
                ha = 'right', 
                va = 'center')
            # if not ax.is_last_row():
            #     plt.setp(ax.get_xticklabels(), visible=False)
            # else:
            plt.setp(
                ax.get_xticklabels(),
                rotation = 90,
                fontsize = 5)
                
        plt.close()
        
    def saveplot(self, destination):
        self.fig.savefig(destination, format="png", bbox_inches="tight")
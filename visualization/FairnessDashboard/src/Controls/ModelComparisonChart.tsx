import React from "react";
import { AccessibleChart, ChartBuilder, IPlotlyProperty, PlotlyMode, SelectionContext } from "mlchartlib";
import { IFairnessContext } from "../IFairnessContext";
import _ from "lodash";
import { MetricsCache } from "../MetricsCache";
import { IAccuracyPickerProps, IParityPickerProps, IFeatureBinPickerProps } from "../FairnessWizard";
import { ParityModes } from "../ParityMetrics";
import { Stack, StackItem } from "office-ui-fabric-react/lib/Stack";
import { Text } from "office-ui-fabric-react/lib/Text";
import { localization } from "../Localization/localization";
import { Spinner, SpinnerSize } from "office-ui-fabric-react/lib/Spinner";
import { mergeStyleSets } from "@uifabric/styling";
import { ActionButton } from "office-ui-fabric-react/lib/Button";
import { AccuracyOptions } from "../AccuracyMetrics";

export interface IModelComparisonProps {
    dashboardContext: IFairnessContext;
    selections: SelectionContext;
    metricsCache: MetricsCache;
    modelCount: number;
    accuracyPickerProps: IAccuracyPickerProps;
    parityPickerProps: IParityPickerProps;
    featureBinPickerProps: IFeatureBinPickerProps;
    onEditConfigs: () => void;
}

export interface IState {
    accuracyArray?: number[];
    disparityArray?: number[];
}

export class ModelComparisonChart extends React.PureComponent<IModelComparisonProps, IState> {
    private readonly plotlyProps: IPlotlyProperty = {
        config: { displaylogo: false, responsive: true, modeBarButtonsToRemove: ['toggleSpikelines', 'hoverClosestCartesian', 'hoverCompareCartesian', 'lasso2d', 'select2d'] },
        data: [
            {
                datapointLevelAccessors: {
                    customdata: {
                        path: ['index'],
                        plotlyPath: 'customdata'
                    }
                },
                mode: PlotlyMode.markers,
                marker: {
                    size: 14
                },
                type: 'scatter',
                xAccessor: 'Accuracy',
                yAccessor: 'Parity'
            }
        ],
        layout: {
            autosize: true,
            font: {
                size: 10
            },
            margin: {
                t: 10
            },
            hovermode: 'closest',
            xaxis: {
                automargin: true,
                title:{
                    text: 'Error'
                }
            },
            yaxis: {
                automargin: true,
                title:{
                    text: 'Disparity'
                }
            },
        } as any
    };

    private static readonly classNames = mergeStyleSets({
        frame: {
            height: "100%"
        },
        spinner: {
            margin: "auto",
            padding: "40px"
        },
        header: {
            backgroundColor: "#EBEBEB",
            padding: "0 90px",
            height: "90px",
            display: "inline-flex",
            flexDirection: "row",
            justifyContent: "space-between",
            alignItems: "center"
        },
        headerTitle: {
            color: "#333333",
            fontSize: "32px",
            lineHeight: "39px",
            fontWeight: "100"
        },
        editButton: {
            color: "#333333",
            fontSize: "12px",
            lineHeight: "20px",
            fontWeight: "400"
        },
        main: {
            height: "100%",
            flex: 1,
            display: "inline-flex",
            flexDirection: "row"
        },
        mainRight: {
            padding: "30px 90px 0 35px",
            width: "300px"
        },
        rightTitle: {
            color: "#333333",
            fontSize: "15px",
            lineHeight: "16px",
            fontWeight: "500",
            paddingBottom: "18px",
            borderBottom: "1px solid #CCCCCC"
        },
        rightText: {
            paddingTop: "16px",
            color: "#333333",
            fontSize: "15px",
            lineHeight: "18px",
            fontWeight: "400",
            paddingBottom: "30px",
            borderBottom: "0.5px dashed #CCCCCC"
        },
        insights: {
            textTransform: "uppercase",
            color: "#333333",
            fontSize: "15px",
            lineHeight: "16px",
            fontWeight: "500",
            padding: "18px 0",
        },
        insightsText: {
            color: "#333333",
            fontSize: "15px",
            lineHeight: "16px",
            fontWeight: "400",
            paddingBottom: "18px",
            borderBottom: "1px solid #CCCCCC"
        },
        chart: {
            padding: "60px 0 100px 35px",
            flex: 1
        }
    });

    public render(): React.ReactNode {
        if (!this.state || this.state.accuracyArray === undefined || this.state.disparityArray === undefined) {
            this.loadData();
            return (
                <Spinner className={ModelComparisonChart.classNames.spinner} size={SpinnerSize.large} label={localization.calculating}/>
            );
        }
        const data = this.state.accuracyArray.map((accuracy, index) => {

            return {
                Parity: this.state.disparityArray[index],
                Accuracy: accuracy,
                index: index
            };
        });
        
        const props = _.cloneDeep(this.plotlyProps);
        props.data = ChartBuilder.buildPlotlySeries(props.data[0], data).map(series => {
            series.name = this.props.dashboardContext.modelNames[series.name];
            return series;
        });
        props.layout.xaxis.title = AccuracyOptions[this.props.accuracyPickerProps.selectedAccuracyKey].title;
        return (
            <Stack className={ModelComparisonChart.classNames.frame}>
                <div className={ModelComparisonChart.classNames.header}>
                    <h2 className={ModelComparisonChart.classNames.headerTitle}>{localization.ModelComparison.title}</h2>
                    <ActionButton iconProps={{iconName: "Edit"}} onClick={this.props.onEditConfigs} className={ModelComparisonChart.classNames.editButton}>{localization.Report.editConfiguration}</ActionButton>
                </div>
                <div className={ModelComparisonChart.classNames.main}>
                    <div className={ModelComparisonChart.classNames.chart}>
                        <AccessibleChart
                            plotlyProps={props}
                            sharedSelectionContext={this.props.selections}
                            theme={undefined}
                        />
                    </div>
                    <div className={ModelComparisonChart.classNames.mainRight}>
                        <div className={ModelComparisonChart.classNames.rightTitle}>{localization.ModelComparison.howToRead}</div>
                        <div className={ModelComparisonChart.classNames.rightText}>{localization.loremIpsum}</div>
                        <div className={ModelComparisonChart.classNames.insights}>{localization.ModelComparison.insights}</div>
                        <div className={ModelComparisonChart.classNames.insightsText}>{localization.loremIpsum}</div>
                    </div>
                </div>
            </Stack>);
    }

    private async loadData(): Promise<void> {
        try {
            const accuracyPromises = new Array(this.props.modelCount).fill(0).map((unused, modelIndex) => {
                return this.props.metricsCache.getMetric(
                    this.props.dashboardContext.binVector,
                    this.props.featureBinPickerProps.selectedBinIndex,
                    modelIndex,
                    this.props.accuracyPickerProps.selectedAccuracyKey);
            });
            const disparityPromises = new Array(this.props.modelCount).fill(0).map((unused, modelIndex) => {
                return this.props.metricsCache.getDisparityMetric(
                    this.props.dashboardContext.binVector,
                    this.props.featureBinPickerProps.selectedBinIndex,
                    modelIndex, this.props.parityPickerProps.selectedParityKey,
                    ParityModes.difference);
            });

            const accuracyArray = (await Promise.all(accuracyPromises)).map(metric => metric.global);
            const disparityArray = await Promise.all(disparityPromises);
            this.setState({accuracyArray, disparityArray});
        } catch {
            // todo;
        }
    }
    // TODO: Reuse if multiselect re-enters design
    // private readonly applySelections = (chartId: string, selectionIds: string[], plotlyProps: IPlotlyProperty) => {
    //     if (!plotlyProps.data || plotlyProps.data.length === 0) {
    //         return;
    //     }
    //     const customData: string[] = (plotlyProps.data[0] as any).customdata;
    //     if (!customData) {
    //         return;
    //     }
    //     const colors = customData.map(modelIndex => {
    //         const selectedIndex = this.props.selections.selectedIds.indexOf(modelIndex);
    //         if (selectedIndex !== -1) {
    //             return FabricStyles.plotlyColorPalette[selectedIndex % FabricStyles.plotlyColorPalette.length];
    //         }
    //         return "#111111";
    //     });
    //     const shapes = customData.map(modelIndex => {
    //         const selectedIndex = this.props.selections.selectedIds.indexOf(modelIndex);
    //         if (selectedIndex !== -1) {
    //             return 1
    //         }
    //         return 0;
    //     });
    //     Plotly.restyle(chartId, 'marker.color' as any, [colors] as any);
    //     Plotly.restyle(chartId, 'marker.symbol' as any, [shapes] as any);
    // }
}
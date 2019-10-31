import React from "react";
import { AccessibleChart, ChartBuilder, IPlotlyProperty, PlotlyMode, SelectionContext } from "mlchartlib";
import { IFairnessContext } from "../IFairnessContext";
import _ from "lodash";
import { MetricsCache } from "../MetricsCache";
import { IAccuracyPickerProps, IParityPickerProps, IFeatureBinPickerProps } from "../FairnessWizard";
import { ParityModes } from "../ParityMetrics";
import { Stack, StackItem } from "office-ui-fabric-react/lib/Stack";
import { ChoiceGroup, IChoiceGroupOption } from 'office-ui-fabric-react/lib/ChoiceGroup';
import { localization } from "../Localization/localization";
import { Spinner, SpinnerSize } from "office-ui-fabric-react/lib/Spinner";
import { mergeStyleSets } from "@uifabric/styling";
import { ActionButton } from "office-ui-fabric-react/lib/Button";
import { AccuracyOptions } from "../AccuracyMetrics";
import { FormatMetrics } from "../FormatMetrics";
import { PredictionTypes } from "../IFairnessProps";

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
    disparityInOutcomes: boolean;
}

export class ModelComparisonChart extends React.PureComponent<IModelComparisonProps, IState> {
    private readonly plotlyProps: IPlotlyProperty = {
        config: { displaylogo: false, responsive: true, modeBarButtonsToRemove: ['toggleSpikelines', 'hoverClosestCartesian', 'hoverCompareCartesian', 'zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'] },
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
                yAccessor: 'Parity',
                hoverinfo: 'none'
            }
        ],
        layout: {
            autosize: true,
            font: {
                size: 10
            },
            margin: {
                t: 4,
                r:0
            },
            hovermode: 'closest',
            xaxis: {
                automargin: true,
                fixedrange: true,
                mirror: true,
                linecolor: '#CCCCCC',
                linewidth: 1,
                title:{
                    text: 'Error'
                }
            },
            yaxis: {
                automargin: true,
                fixedrange: true,
                title:{
                    text: 'Disparity'
                }
            },
        } as any
    };

    private static readonly classNames = mergeStyleSets({
        frame: {
            flex: 1,
            display: "flex",
            flexDirection: "column"
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
            padding: "30px 0 0 35px",
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
            padding: "16px 15px 30px 0",
            color: "#333333",
            fontSize: "15px",
            lineHeight: "18px",
            fontWeight: "400",
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
            paddingRight: "15px",
            borderBottom: "1px solid #CCCCCC"
        },
        chart: {
            padding: "60px 0 0 0",
            flex: 1
        },
        textSection: {
            paddingBottom: "5px"
        },
        radio: {
            paddingBottom: "30px",
            paddingLeft: "75px"
        }
    });

    constructor(props: IModelComparisonProps) {
        super(props);
        this.state = {
            disparityInOutcomes: true
        };
    }

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

        let minAccuracy: number = Number.MAX_SAFE_INTEGER;
        let maxAccuracy: number = Number.MIN_SAFE_INTEGER;
        let maxDisparity: number = Number.MIN_SAFE_INTEGER;
        let minDisparity: number = Number.MAX_SAFE_INTEGER;
        let minAccuracyIndex: number;
        let maxAccuracyIndex: number;
        let minDisparityIndex: number;
        let maxDisparityIndex: number;
        this.state.accuracyArray.forEach((value, index) => {
            if (value >= maxAccuracy) {
                maxAccuracyIndex = index;
                maxAccuracy = value;
            }
            if (value <= minAccuracy) {
                minAccuracyIndex = index;
                minAccuracy = value;
            }
        });
        this.state.disparityArray.forEach((value, index) => {
            if (value >= maxDisparity) {
                maxDisparityIndex = index;
                maxDisparity = value;
            }
            if (value <= minDisparity) {
                minDisparityIndex = index;
                minDisparity = value;
            }
        });
        const formattedMinAccuracy = FormatMetrics.formatNumbers(minAccuracy, this.props.accuracyPickerProps.selectedAccuracyKey);
        const formattedMaxAccuracy = FormatMetrics.formatNumbers(maxAccuracy, this.props.accuracyPickerProps.selectedAccuracyKey);
        const formattedMinDisparity = FormatMetrics.formatNumbers(minDisparity, this.props.accuracyPickerProps.selectedAccuracyKey);
        const formattedMaxDisparity = FormatMetrics.formatNumbers(maxDisparity, this.props.accuracyPickerProps.selectedAccuracyKey);
        const selectedMetric = AccuracyOptions[this.props.accuracyPickerProps.selectedAccuracyKey];
        const insights2 = localization.formatString(
            localization.ModelComparison.insightsText2,
            selectedMetric.title,
            formattedMinAccuracy,
            formattedMaxAccuracy,
            formattedMinDisparity,
            formattedMaxDisparity
        );
        const insights3 = localization.formatString(
            localization.ModelComparison.insightsText3,
            selectedMetric.title.toLowerCase(),
            selectedMetric.isMinimization ? formattedMinAccuracy : formattedMaxAccuracy, 
            FormatMetrics.formatNumbers(this.state.disparityArray[selectedMetric.isMinimization ? minAccuracyIndex : maxAccuracyIndex], this.props.accuracyPickerProps.selectedAccuracyKey)
        );

        const insights4 = localization.formatString(
            localization.ModelComparison.insightsText4,
            selectedMetric.title.toLowerCase(),
            FormatMetrics.formatNumbers(this.state.accuracyArray[minDisparityIndex], this.props.accuracyPickerProps.selectedAccuracyKey),
            formattedMinDisparity
        );

        const howToReadText = localization.formatString(
            localization.ModelComparison.howToReadText,
            this.props.modelCount.toString(),
            selectedMetric.title.toLowerCase(),
            selectedMetric.isMinimization ? localization.ModelComparison.lower : localization.ModelComparison.higher
        );
        
        const props = _.cloneDeep(this.plotlyProps);
        props.data = ChartBuilder.buildPlotlySeries(props.data[0], data).map(series => {
            series.name = this.props.dashboardContext.modelNames[series.name];
            return series;
        });
        const accuracyMetricTitle = AccuracyOptions[this.props.accuracyPickerProps.selectedAccuracyKey].title 
        props.layout.xaxis.title = accuracyMetricTitle;
        props.layout.yaxis.title = this.state.disparityInOutcomes ? localization.ModelComparison.disparityInOutcomes :
            localization.formatString(localization.ModelComparison.disparityInAccuracy, accuracyMetricTitle.toLowerCase()) as string
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
                        <div className={ModelComparisonChart.classNames.rightText}>{howToReadText}</div>
                        <div className={ModelComparisonChart.classNames.insights}>{localization.ModelComparison.insights}</div>
                        <div className={ModelComparisonChart.classNames.insightsText}>
                            <div className={ModelComparisonChart.classNames.textSection}>{insights2}</div>
                            <div className={ModelComparisonChart.classNames.textSection}>{insights3}</div>
                            <div>{insights4}</div>
                        </div>
                    </div>
                </div>
                <div>
                    <ChoiceGroup
                        className={ModelComparisonChart.classNames.radio}
                        selectedKey={this.state.disparityInOutcomes ? "outcomes" : "accuracy"}
                        options={[
                            {
                            key: 'accuracy',
                            text: localization.formatString(localization.ModelComparison.disparityInAccuracy, 
                                AccuracyOptions[this.props.accuracyPickerProps.selectedAccuracyKey].title.toLowerCase()) as string
                            },
                            {
                            key: 'outcomes',
                            text: localization.ModelComparison.disparityInOutcomes
                            }
                        ]}
                        onChange={this.disparityChanged}
                        label={localization.ModelComparison.howToMeasureDisparity}
                        required={false}
                        ></ChoiceGroup>
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
            const disparityMetric = this.state.disparityInOutcomes ?
                (this.props.dashboardContext.modelMetadata.predictionType === PredictionTypes.binaryClassification ?
                    "selection_rate" : "average") :
                this.props.accuracyPickerProps.selectedAccuracyKey;
            const disparityPromises = new Array(this.props.modelCount).fill(0).map((unused, modelIndex) => {
                return this.props.metricsCache.getDisparityMetric(
                    this.props.dashboardContext.binVector,
                    this.props.featureBinPickerProps.selectedBinIndex,
                    modelIndex, 
                    disparityMetric,
                    ParityModes.difference);
            });

            const accuracyArray = (await Promise.all(accuracyPromises)).map(metric => metric.global);
            const disparityArray = await Promise.all(disparityPromises);
            this.setState({accuracyArray, disparityArray});
        } catch {
            // todo;
        }
    }

    private readonly disparityChanged = (ev: React.FormEvent<HTMLInputElement>, option: IChoiceGroupOption): void => {
        const disparityInOutcomes = option.key !== "accuracy";
        if (this.state.disparityInOutcomes !== disparityInOutcomes) {
            this.setState({disparityInOutcomes, disparityArray: undefined});
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
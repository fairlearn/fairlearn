import React from "react";
import { Stack, StackItem } from "office-ui-fabric-react/lib/Stack";
import { IModelComparisonProps } from "./Controls/ModelComparisonChart";
import { Text } from "office-ui-fabric-react/lib/Text";
import { localization } from "./Localization/localization";
import { Separator } from "office-ui-fabric-react/lib/Separator";
import { Spinner, SpinnerSize } from 'office-ui-fabric-react/lib/Spinner';
import { mergeStyleSets } from "@uifabric/styling";
import _ from "lodash";
import { ParityModes } from "./ParityMetrics";
import { IPlotlyProperty, AccessibleChart } from "mlchartlib";
import { ActionButton } from "office-ui-fabric-react/lib/Button";
import { SummaryTable } from "./Controls/SummaryTable";
import { PredictionTypes, IMetricResponse } from "./IFairnessProps";
import { AccuracyOptions } from "./AccuracyMetrics";
import { NONAME } from "dns";
import { ChartColors } from "./ChartColors";

interface IMetrics {
    globalAccuracy: number;
    binnedAccuracy: number[];
    accuracyDisparity: number;
    globalOutcome: number;
    outcomeDisparity: number;
    binnedOutcome: number[];
    // Optional, based on model type
    binnedOverprediction?: number[];
    binnedUnderprediction?: number[];
    // different length, raw unbinned errors and predictions
    errors?: number[];
    predictions?: number[];
}

export interface IState {
    metrics?: IMetrics;
}

export interface IReportProps extends IModelComparisonProps {
    selectedModelIndex: number;
}

export class WizardReport extends React.PureComponent<IReportProps, IState> {
    private static separatorStyle = {
        root: [{
          selectors: {
            '::after': {
              backgroundColor: 'darkgrey',
            },
          }
        }]
    };

    private static readonly classNames = mergeStyleSets({
        spinner: {
            margin: "auto",
            padding: "40px"
        },
        header: {
            padding: "0 90px",
            backgroundColor: "#F2F2F2"
        },
        multimodelButton: {
            marginTop: "20px",
            padding: 0,
            color: "#333333",
            fontSize: "12px",
            lineHeight: "16px",
            fontWeight: "400"
        },
        headerTitle: {
            paddingTop: "10px",
            color: "#333333",
            fontSize: "32px",
            lineHeight: "39px",
            fontWeight: "100"
        },
        headerBanner: {
            display: "flex"
        },
        bannerWrapper: {
            width: "100%",
            paddingTop: "18px",
            paddingBottom: "15px",
            display: "inline-flex",
            flexDirection: "row",
            justifyContent: "space-between"
        },
        editButton: {
            color: "#333333",
            fontSize: "12px",
            lineHeight: "20px",
            fontWeight: "400"
        },
        metricText: {
            color: "#333333",
            fontSize: "36px",
            lineHeight: "44px",
            fontWeight: "100",
            paddingRight: "12px"
        },
        firstMetricLabel: {
            color: "#333333",
            fontSize: "12px",
            lineHeight: "16px",
            fontWeight: "400",
            padding: "0 17px 0 12px",
            alignSelf: "center",
            maxWidth: "90px",
            borderRight: "1px solid #CCCCCC",
            marginRight: "20px"
        },
        metricLabel: {
            color: "#333333",
            fontSize: "12px",
            lineHeight: "16px",
            fontWeight: "400",
            alignSelf: "center",
            maxWidth: "130px"
        },
        presentationArea: {
            display: "flex",
            flexDirection: "row",
            padding: "20px 0 30px 90px"
        },
        chartWrapper: {
            flex: "1 0 40%",
            paddingTop: "23px"
        },
        mainRight: {
            minWidth: "200px",
            paddingLeft: "35px",
            flexBasis: "300px",
            flexShrink: 1
        },
        rightTitle: {
            color: "#333333",
            fontSize: "15px",
            lineHeight: "16px",
            fontWeight: "500",
            paddingBottom: "11px",
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
        tableWrapper: {
            paddingBottom: "20px"
        },
        textRow: {
            display: "flex",
            flexDirection: "row",
            alignItems: "center",
            paddingBottom: "7px"
        },
        colorBlock: {
            width: "15px",
            height: "15px",
            marginRight: "9px"
        },
        multimodelSection: {
            display: "flex",
            flexDirection:"row"
        },
        modelLabel: {
            alignSelf: "center",
            paddingLeft: "35px",
            paddingTop: "16px",
            color: "#333333",
            fontSize: "26px",
            lineHeight: "16px",
            fontWeight: "400"
        }
    });

    private static barPlotlyProps: IPlotlyProperty = {
        config: { displaylogo: false, responsive: true, modeBarButtonsToRemove: ['toggleSpikelines', 'hoverClosestCartesian', 'hoverCompareCartesian', 'zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'] },
        data: [
            {
                orientation: 'h',
                type: 'bar'
            }
        ],
        layout: {
            autosize: true,
            barmode: 'relative',
            font: {
                size: 10
            },
            margin: {
                t: 4,
                l: 0,
                r: 0,
                b: 20
            },
            showlegend: false,
            hovermode: 'closest',
            plot_bgcolor: "#FAFAFA",
            xaxis: {
                fixedrange: true,
                autorange: true,
                mirror: true,
                linecolor: '#CCCCCC',
                linewidth: 1,
            },
            yaxis: {
                fixedrange: true,
                showticklabels: false,
                showgrid: true,
                dtick: 1,
                tick0: 0.5,
                gridcolor: '#CCCCCC',
                gridwidth: 1,
                autorange: "reversed"
            }
        } as any
    };

    render(): React.ReactNode {
        if (!this.state || !this.state.metrics) {
            this.loadData();
            return (
                <Spinner className={WizardReport.classNames.spinner} size={SpinnerSize.large} label={localization.calculating}/>
            );
        }

        const alternateHeight = this.props.featureBinPickerProps.featureBins.length * 60 + 106;
        const areaHeights = Math.max(460, alternateHeight);

        const accuracyKey = this.props.accuracyPickerProps.selectedAccuracyKey;
        const outcomeKey = this.props.dashboardContext.modelMetadata.predictionType === PredictionTypes.binaryClassification ? "selection_rate" : "average";
        const outcomeMetric = AccuracyOptions[outcomeKey];

        const accuracyPlot = _.cloneDeep(WizardReport.barPlotlyProps);
        const opportunityPlot = _.cloneDeep(WizardReport.barPlotlyProps);
        const nameIndex = this.props.dashboardContext.groupNames.map((unuxed, i) => i);
        let howToReadAccuracySection: React.ReactNode;
        let insightsAccuracySection: React.ReactNode;
        let howToReadOutcomesSection: React.ReactNode;
        let insightsOutcomesSection: React.ReactNode;

        if (this.props.dashboardContext.modelMetadata.predictionType === PredictionTypes.binaryClassification) {
            accuracyPlot.data = [
                {
                    x: this.state.metrics.binnedOverprediction,
                    y: nameIndex,
                    text: this.state.metrics.binnedOverprediction.map(num => (num as number).toLocaleString(undefined, {style: "percent", maximumSignificantDigits: 2})),
                    name: localization.Metrics.overprediction,
                    width: 0.5,
                    color: ChartColors[0],
                    orientation: 'h',
                    type: 'bar',
                    textposition: 'auto',
                    hoverinfo: "skip"
                } as any, {
                    x: this.state.metrics.binnedUnderprediction.map(x => -1 * x),
                    y: nameIndex,
                    text: this.state.metrics.binnedUnderprediction.map(num => (num as number).toLocaleString(undefined, {style: "percent", maximumSignificantDigits: 2})),
                    name: localization.Metrics.underprediction,
                    width: 0.5,
                    color: ChartColors[1],
                    orientation: 'h',
                    type: 'bar',
                    textposition: 'auto',
                    hoverinfo: "skip"
                }
            ];
            accuracyPlot.layout.annotations = [
                {
                    text: localization.Report.underestimationError,
                    x: 0.02,
                    y: 1,
                    yref: 'paper', xref: 'paper',
                    showarrow: false,
                    font: {color:'#666666', size: 10}
                },
                    {
                    text: localization.Report.overestimationError,
                    x: 0.98,
                    y: 1,
                    yref: 'paper', xref: 'paper',
                    showarrow: false,
                    font: {color:'#666666', size: 10}
                },
                {
                    text: localization.Report.underpredictionExplanation,
                    x: 0.02,
                    y: 0.97,
                    yref: 'paper', xref: 'paper',
                    showarrow: false,
                    font: {color:'#666666', size: 10}
                },
                    {
                    text: localization.Report.overpredictionExplanation,
                    x: 0.98,
                    y: 0.97,
                    yref: 'paper', xref: 'paper',
                    showarrow: false,
                    font: {color:'#666666', size: 10}
                }
            ];
            accuracyPlot.layout.xaxis.tickformat = ',.0%';
            opportunityPlot.data = [
                {
                    x: this.state.metrics.binnedOutcome,
                    y: nameIndex,
                    text: this.state.metrics.binnedOutcome.map(num => (num as number).toLocaleString(undefined, {style: "percent", maximumSignificantDigits: 2})),
                    name: outcomeMetric.title,
                    color: ChartColors[0],
                    orientation: 'h',
                    type: 'bar',
                    textposition: 'auto',
                    hoverinfo: "skip"
                } as any
            ];
            opportunityPlot.layout.xaxis.tickformat = ',.0%';
            howToReadAccuracySection = (<div>
                <div className={WizardReport.classNames.textRow}>
                    <div className={WizardReport.classNames.colorBlock} style={{backgroundColor: ChartColors[0]}}/>
                    <div>{localization.Report.overestimationError}</div>
                </div>
                <div className={WizardReport.classNames.textRow}>
                    <div className={WizardReport.classNames.colorBlock} style={{backgroundColor: ChartColors[1]}}/>
                    <div>{localization.Report.underestimationError}</div>
                </div>
                <div className={WizardReport.classNames.textRow}>{localization.Report.classificationAccuracyHowToRead}</div>
            </div>);
            howToReadOutcomesSection = (<div>
                <div className={WizardReport.classNames.textRow}>{localization.Report.classificationOutcomesHowToRead}</div>
            </div>);
        } if (this.props.dashboardContext.modelMetadata.predictionType === PredictionTypes.probability) {
            accuracyPlot.data = [
                {
                    x: this.state.metrics.binnedOverprediction,
                    y: nameIndex,
                    text: this.state.metrics.binnedOverprediction.map(num => (num as number).toLocaleString(undefined, {style: "percent", maximumSignificantDigits: 2})),
                    name: localization.Metrics.overprediction,
                    width: 0.5,
                    color: ChartColors[0],
                    orientation: 'h',
                    type: 'bar',
                    textposition: 'auto',
                    hoverinfo: "skip"
                } as any, {
                    x: this.state.metrics.binnedUnderprediction.map(x => -1 * x),
                    y: nameIndex,
                    text: this.state.metrics.binnedUnderprediction.map(num => (num as number).toLocaleString(undefined, {style: "percent", maximumSignificantDigits: 2})),
                    name: localization.Metrics.underprediction,
                    width: 0.5,
                    color: ChartColors[1],
                    orientation: 'h',
                    type: 'bar',
                    textposition: 'auto',
                    hoverinfo: "skip"
                }
            ];
            accuracyPlot.layout.annotations = [
                {
                    text: localization.Report.underestimationError,
                    x: 0.1,
                    y: 1,
                    yref: 'paper', xref: 'paper',
                    showarrow: false,
                    font: {color:'#666666', size: 10}
                },
                {
                    text: localization.Report.overestimationError,
                    x: 0.9,
                    y: 1,
                    yref: 'paper', xref: 'paper',
                    showarrow: false,
                    font: {color:'#666666', size: 10}
                }
            ];
            opportunityPlot.data = [
                {
                    x: this.state.metrics.predictions,
                    y: this.props.dashboardContext.binVector,
                    type: 'box',
                    color: ChartColors[0],
                    boxmean: true,
                    orientation: 'h',
                    boxpoints: 'all',
                    jitter: 0.4,
                    pointpos: 0,
                } as any
            ];
            howToReadAccuracySection = (<div>
                <div className={WizardReport.classNames.textRow}>
                    <div className={WizardReport.classNames.colorBlock} style={{backgroundColor: ChartColors[0]}}/>
                    <div>{localization.Report.overestimationError}</div>
                </div>
                <div className={WizardReport.classNames.textRow}>
                    <div className={WizardReport.classNames.colorBlock} style={{backgroundColor: ChartColors[1]}}/>
                    <div>{localization.Report.underestimationError}</div>
                </div>
                <div className={WizardReport.classNames.textRow}>{localization.Report.probabilityAccuracyHowToRead}</div>
            </div>);
            howToReadOutcomesSection = (<div>
                <div className={WizardReport.classNames.textRow}>{localization.Report.regressionOutcomesHowToRead}</div>
            </div>);
        } if (this.props.dashboardContext.modelMetadata.predictionType === PredictionTypes.regression) {
            accuracyPlot.data = [
                {
                    x: this.state.metrics.errors,
                    y: this.props.dashboardContext.binVector,
                    type: 'box',
                    color: ChartColors[0],
                    orientation: 'h',
                    boxmean: true,
                    boxpoints: 'all',
                    jitter: 0.4,
                    pointpos: 0,
                } as any
            ];
            opportunityPlot.data = [
                {
                    x: this.state.metrics.predictions,
                    y: this.props.dashboardContext.binVector,
                    type: 'box',
                    color: ChartColors[0],
                    boxmean: true,
                    orientation: 'h',
                    boxpoints: 'all',
                    jitter: 0.4,
                    pointpos: 0,
                } as any
            ];
            howToReadAccuracySection = (<div>
                <div className={WizardReport.classNames.textRow}>{localization.Report.regressionAccuracyHowToRead}</div>
            </div>);
            howToReadOutcomesSection = (<div>
                <div className={WizardReport.classNames.textRow}>{localization.Report.regressionOutcomesHowToRead}</div>
            </div>);
        }
        
        const globalAccuracyString = this.formatNumbers(this.state.metrics.globalAccuracy, accuracyKey);
        const disparityAccuracyString = this.formatNumbers(this.state.metrics.accuracyDisparity, accuracyKey);
        
        const globalOutcomeString = this.formatNumbers(this.state.metrics.globalOutcome, outcomeKey);
        const disparityOutcomeString = this.formatNumbers(this.state.metrics.outcomeDisparity, outcomeKey);
        const formattedBinAccuracyValues = this.state.metrics.binnedAccuracy.map(value => 
            this.formatNumbers(value, accuracyKey));
        const formattedBinOutcomeValues = this.state.metrics.binnedOutcome.map(value => 
            this.formatNumbers(value, outcomeKey));
        return (<div style={{height: "100%", overflowY:"auto"}}>
            <div className={WizardReport.classNames.header}>
                {this.props.modelCount > 1 &&
                    <div className={WizardReport.classNames.multimodelSection}>
                        <ActionButton
                            className={WizardReport.classNames.multimodelButton}
                            iconProps={{iconName: "ChevronLeft"}}
                            onClick={this.clearModelSelection}>
                            {localization.Report.backToComparisons}
                        </ActionButton>
                        <div className={WizardReport.classNames.modelLabel}>
                            {localization.formatString(localization.Report.modelName, this.props.selectedModelIndex)}
                        </div>
                    </div>}
                <div className={WizardReport.classNames.headerTitle}>{localization.Report.title}</div>
                <div className={WizardReport.classNames.bannerWrapper}>
                    <div className={WizardReport.classNames.headerBanner}>
                        <div className={WizardReport.classNames.metricText}>{globalAccuracyString}</div>
                        <div className={WizardReport.classNames.firstMetricLabel}>{localization.formatString(localization.Report.globalAccuracyText, AccuracyOptions[accuracyKey].title.toLowerCase())}</div>
                        <div className={WizardReport.classNames.metricText}>{disparityAccuracyString}</div>
                        <div className={WizardReport.classNames.metricLabel}>{localization.formatString(localization.Report.accuracyDisparityText, AccuracyOptions[accuracyKey].title.toLowerCase())}</div>
                    </div>
                    <ActionButton
                        className={WizardReport.classNames.editButton}
                        iconProps={{iconName: "Edit"}}
                        onClick={this.onEditConfigs}>{localization.Report.editConfiguration}</ActionButton>
                </div>
            </div>
            <div className={WizardReport.classNames.presentationArea} style={{height: `${areaHeights}px`}}>
                    <SummaryTable 
                        binLabels={this.props.dashboardContext.groupNames}
                        formattedBinValues={formattedBinAccuracyValues}
                        metricLabel={AccuracyOptions[accuracyKey].title}
                        binValues={this.state.metrics.binnedAccuracy}/>
                    <div className={WizardReport.classNames.chartWrapper}>
                        <AccessibleChart
                            plotlyProps={accuracyPlot}
                            sharedSelectionContext={undefined}
                            theme={undefined}
                        />
                    </div>
                    <div className={WizardReport.classNames.mainRight}>
                        <div className={WizardReport.classNames.rightTitle}>{localization.ModelComparison.howToRead}</div>
                        <div className={WizardReport.classNames.rightText}>{howToReadAccuracySection}</div>
                        {/* <div className={WizardReport.classNames.insights}>{localization.ModelComparison.insights}</div>
                        <div className={WizardReport.classNames.insightsText}>{localization.loremIpsum}</div> */}
                    </div>
            </div>
            <div className={WizardReport.classNames.header}>
                <div className={WizardReport.classNames.headerTitle}>{localization.Report.outcomesTitle}</div>
                <div className={WizardReport.classNames.bannerWrapper}>
                    <div className={WizardReport.classNames.headerBanner}>
                        <div className={WizardReport.classNames.metricText}>{globalOutcomeString}</div>
                        <div className={WizardReport.classNames.firstMetricLabel}>{localization.formatString(localization.Report.globalAccuracyText, outcomeMetric.title.toLowerCase())}</div>
                        <div className={WizardReport.classNames.metricText}>{disparityOutcomeString}</div>
                        <div className={WizardReport.classNames.metricLabel}>{localization.formatString(localization.Report.accuracyDisparityText, outcomeMetric.title.toLowerCase())}</div>
                    </div>
                </div>
            </div>
            <div className={WizardReport.classNames.presentationArea} style={{height: `${areaHeights}px`}}>
                    <SummaryTable 
                        binLabels={this.props.dashboardContext.groupNames}
                        formattedBinValues={formattedBinOutcomeValues}
                        metricLabel={outcomeMetric.title}
                        binValues={this.state.metrics.binnedOutcome}/>
                    <div className={WizardReport.classNames.chartWrapper}>
                        <AccessibleChart
                            plotlyProps={opportunityPlot}
                            sharedSelectionContext={undefined}
                            theme={undefined}
                        />
                    </div>
                    <div className={WizardReport.classNames.mainRight}>
                        <div className={WizardReport.classNames.rightTitle}>{localization.ModelComparison.howToRead}</div>
                        <div className={WizardReport.classNames.rightText}>{howToReadOutcomesSection}</div>
                        {/* <div className={WizardReport.classNames.insights}>{localization.ModelComparison.insights}</div>
                        <div className={WizardReport.classNames.insightsText}>{localization.loremIpsum}</div> */}
                    </div>
            </div>
        </div>);
    }

    private readonly formatNumbers = (value: number, key: string, isRatio: boolean = false): string => {
        if (value === null || value === undefined) {
            return NaN.toString();
        }
        const styleObject = {maximumSignificantDigits: 3};
        if (AccuracyOptions[key].isPercentage && !isRatio) {
            (styleObject as any).style = "percent";
        }
        return value.toLocaleString(undefined, styleObject);
    }

    private readonly clearModelSelection = (): void => {
        this.props.selections.onSelect([]);
    }

    private readonly onEditConfigs = (): void => {
        if (this.props.modelCount > 1) {
            this.props.selections.onSelect([]);
        }
        this.props.onEditConfigs();
    }

    private async loadData(): Promise<void> {
        try {
            let binnedFNR: number[];
            let binnedFPR: number[];
            let binnedOverprediction: number[];
            let binnedUnderprediction: number[];
            let predictions: number[];
            let errors: number[];
            let outcomes: IMetricResponse;
            let outcomeDisparity: number;
            const accuracy = (await this.props.metricsCache.getMetric(
                this.props.dashboardContext.binVector,
                this.props.featureBinPickerProps.selectedBinIndex, 
                this.props.selectedModelIndex,
                this.props.accuracyPickerProps.selectedAccuracyKey));
            const accuracyDisparity = await this.props.metricsCache.getDisparityMetric(
                this.props.dashboardContext.binVector,
                this.props.featureBinPickerProps.selectedBinIndex, 
                this.props.selectedModelIndex,
                this.props.accuracyPickerProps.selectedAccuracyKey,
                ParityModes.difference);
            if (this.props.dashboardContext.modelMetadata.predictionType === PredictionTypes.binaryClassification) {
                binnedUnderprediction = (await this.props.metricsCache.getMetric(
                    this.props.dashboardContext.binVector,
                    this.props.featureBinPickerProps.selectedBinIndex, 
                    this.props.selectedModelIndex,
                    "underprediction")).bins;
                binnedOverprediction = (await this.props.metricsCache.getMetric(
                    this.props.dashboardContext.binVector,
                    this.props.featureBinPickerProps.selectedBinIndex, 
                    this.props.selectedModelIndex,
                    "overprediction")).bins;
                outcomes = await this.props.metricsCache.getMetric(
                    this.props.dashboardContext.binVector,
                    this.props.featureBinPickerProps.selectedBinIndex, 
                    this.props.selectedModelIndex,
                    "selection_rate");
                outcomeDisparity = await this.props.metricsCache.getDisparityMetric(
                    this.props.dashboardContext.binVector,
                    this.props.featureBinPickerProps.selectedBinIndex, 
                    this.props.selectedModelIndex,
                    "selection_rate",
                    ParityModes.difference);
            } if (this.props.dashboardContext.modelMetadata.predictionType === PredictionTypes.probability) {
                predictions = this.props.dashboardContext.predictions[this.props.selectedModelIndex];
                binnedOverprediction = (await this.props.metricsCache.getMetric(
                    this.props.dashboardContext.binVector,
                    this.props.featureBinPickerProps.selectedBinIndex, 
                    this.props.selectedModelIndex,
                    "overprediction")).bins;
                binnedUnderprediction = (await this.props.metricsCache.getMetric(
                    this.props.dashboardContext.binVector,
                    this.props.featureBinPickerProps.selectedBinIndex, 
                    this.props.selectedModelIndex,
                    "underprediction")).bins;
                outcomes = await this.props.metricsCache.getMetric(
                    this.props.dashboardContext.binVector,
                    this.props.featureBinPickerProps.selectedBinIndex, 
                    this.props.selectedModelIndex,
                    "average");
                outcomeDisparity = await this.props.metricsCache.getDisparityMetric(
                    this.props.dashboardContext.binVector,
                    this.props.featureBinPickerProps.selectedBinIndex, 
                    this.props.selectedModelIndex,
                    "average",
                    ParityModes.difference);
            } if (this.props.dashboardContext.modelMetadata.predictionType === PredictionTypes.regression) {
                predictions = this.props.dashboardContext.predictions[this.props.selectedModelIndex];
                errors = predictions.map((predicted, index) => {
                    return predicted - this.props.dashboardContext.trueY[index];
                });
                outcomes = await this.props.metricsCache.getMetric(
                    this.props.dashboardContext.binVector,
                    this.props.featureBinPickerProps.selectedBinIndex, 
                    this.props.selectedModelIndex,
                    "average");
                outcomeDisparity = await this.props.metricsCache.getDisparityMetric(
                    this.props.dashboardContext.binVector,
                    this.props.featureBinPickerProps.selectedBinIndex, 
                    this.props.selectedModelIndex,
                    "average",
                    ParityModes.difference);
            }
            this.setState({
                metrics: {
                    globalAccuracy: accuracy.global,
                    binnedAccuracy: accuracy.bins,
                    accuracyDisparity,
                    globalOutcome: outcomes.global,
                    binnedOutcome: outcomes.bins,
                    outcomeDisparity,
                    predictions,
                    errors,
                    binnedOverprediction,
                    binnedUnderprediction
                }
            });
        } catch {
            // todo;
        }
    }
}
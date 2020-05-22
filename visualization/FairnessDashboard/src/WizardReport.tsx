import React from "react";
import ReactModal from "react-modal";
import { Stack, StackItem } from "office-ui-fabric-react/lib/Stack";
import { IModelComparisonProps } from "./Controls/ModelComparisonChart";
import { Text } from "office-ui-fabric-react/lib/Text";
import { localization } from "./Localization/localization";
import { Separator } from "office-ui-fabric-react/lib/Separator";
import { Dropdown, DropdownMenuItemType, IDropdownStyles, IDropdownOption } from 'office-ui-fabric-react/lib/Dropdown';
import { Spinner, SpinnerSize } from 'office-ui-fabric-react/lib/Spinner';
import { mergeStyleSets } from "@uifabric/styling";
import _ from "lodash";
import { ParityModes } from "./ParityMetrics";
import { IPlotlyProperty, AccessibleChart } from "mlchartlib";
import { ActionButton } from "office-ui-fabric-react/lib/Button";
import { SummaryTable } from "./Controls/SummaryTable";
import { OverallTable } from "./Controls/OverallTable";
import { PredictionTypes, IMetricResponse } from "./IFairnessProps";
import { AccuracyOptions } from "./AccuracyMetrics";
import { NONAME } from "dns";
import { ChartColors } from "./ChartColors";
import { withSlots } from "office-ui-fabric-react/lib/Foundation";

interface IMetrics {
    globalAccuracy: number;
    binnedAccuracy: number[];
    accuracyDisparity: number;
    globalOutcome: number;
    outcomeDisparity: number;
    binnedOutcome: number[];
    // Optional, based on model type
    globalOverprediction?: number;
    globalUnderprediction?: number;
    binnedOverprediction?: number[];
    binnedUnderprediction?: number[];
    // different length, raw unbinned errors and predictions
    errors?: number[];
    predictions?: number[];
}

export interface IState {
    metrics?: IMetrics;
    featureKey?: string;
    showModalHelp?: boolean;
    expandAttributes: boolean;
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
            padding: "0px 50px 20px 50px",
            backgroundColor: "#222222"
        },
        multimodelButton: {
            marginTop: "20px",
            padding: 0,
            color: "#ffffff",
            fontSize: "12px",
            fontWeight: "400"
        },
        headerTitle: {
            paddingTop: "10px",
            color: "#ffffff",
            fontSize: "30px",
            lineHeight: "43px",
            fontWeight: "300"
        },
        headerBanner: {
            display: "flex"
        },
        headerOptions: {
            backgroundColor: "#222222",
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
            color: "#ffffff",
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
            padding: "8px 12px 0 12px",
            maxWidth: "120px",
            borderRight: "1px solid #CCCCCC",
            marginRight: "20px"
        },
        metricLabel: {
            color: "#333333",
            fontSize: "12px",
            lineHeight: "16px",
            fontWeight: "400",
            paddingTop: "8px",
            maxWidth: "130px"
        },
        expandAttributes: {
            color: "#333333",
            fontSize: "12px",
            lineHeight: "16px",
            fontWeight: "normal",
            height: "26px",
            marginLeft: "50px"
        },
        overallArea: {
            display: "flex",
            flexDirection: "row",
            padding: "20px 0 30px 50px",
            backgroundColor: 'white'
        },
        presentationArea: {
            display: "flex",
            flexDirection: "row",
            padding: "20px 0 30px 50px",
            backgroundColor: 'white'
        },
        chartWrapper: {
            flex: "1 0 40%",
            display: "flex",
            flexDirection: "column"
        },
        chartBody: {
            flex: 1
        },
        chartHeader: {
            height: "23px",
            paddingLeft: "10px",
            color: "#333333",
            fontSize: "12px",
            lineHeight: "12px",
            fontWeight: "500"
        },
        dropDown: {
            margin: "10px 0px",
            display: "inline-block"
        },
        main: {
            display: "flex",
            flexDirection: "row"
        },
        mainLeft: {
            width: "75%",
        },
        mainRight: {
            marginLeft: "20px",
            width: "240px",
            paddingLeft: "35px",
            // flexBasis: "300px",
            // flexShrink: 1,
            backgroundColor: "#f2f2f2",
            boxShadow: "-1px 0px 0px #D2D2D2"
        },
        rightTitle: {
            color: "#333333",
            fontSize: "12px",
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
            fontWeight: "400"
            // borderBottom: "0.5px dashed #CCCCCC"
        },
        insights: {
            // textTransform: "uppercase",
            color: "#333333",
            fontSize: "18px",
            lineHeight: "22px",
            fontWeight: "normal",
            padding: "18px 0",
        },
        insightsText: {
            color: "#333333",
            fontSize: "12px",
            lineHeight: "16px",
            fontWeight: "normal",
            paddingBottom: "18px",
            paddingRight: "15px"
            // borderBottom: "1px solid #CCCCCC"
        },
        downloadReport: {
            color: "#333333",
            fontSize: "12px",
            lineHeight: "16px",
            fontWeight: "normal",
            paddingTop: "20px",
            paddingBottom: "20px",
            paddingLeft: "0px",
            // border: "1px solid #CCCCCC"
        },
        tableWrapper: {
            paddingBottom: "20px"
        },
        textRow: {
            float: "left",
            display: "flex",
            flexDirection: "row",
            alignItems: "center",
            paddingBottom: "7px",
            paddingLeft: "50px",
            paddingRight: "50px"
        },
        infoButton: {
            color: "#999999",
            float: "left",
            width: "15px",
            height: "15px",
            textAlign: "center",
            fontSize: "12px",
            lineHeight: "14px",
            fontWeight: "600",
            borderRadius: "50%",
            border: "1px solid",
            marginTop: "3px",
            marginRight: "3px",
            marginLeft: "250px"
        },
        closeButton: {
            color: "#FFFFFF",
            float: "right",
            fontFamily: "Arial",
            fontSize: "20px",
            lineHeight: "20px",
            fontWeight: "400",
            paddingLeft: "20px"
        },
        doneButton: {
            color: "#FFFFFF",
            borderRadius: "5px",
            background: "#5A53FF",
            padding: "5px 15px",
            selectors: {
                '&:hover': { color: "#ffffff" }
            }
        },
        equalizedOdds: {
            float: "left",
            fontSize: "18px",
            lineHeight: "22px",
            fontWeight: "normal",
            paddingTop: "30px",
            paddingLeft: "50px"
        },
        howTo: {
            paddingTop: "20px",
            paddingLeft: "100px"
        },
        colorBlock: {
            width: "15px",
            height: "15px",
            marginRight: "9px"
        },
        modalContentHelp: {
            float: 'left',
            paddingTop: '10px',
            paddingRight: '20px',
            wordWrap: "break-word",
            width: "300px",
            textAlign: "center"
        },
        multimodelSection: {
            display: "flex",
            flexDirection:"row"
        },
        modelLabel: {
            alignSelf: "center",
            color: "#ffffff",
            fontSize: "26px",
            fontWeight: "400",
            paddingTop: "10px",
            paddingBottom: "10px"
        },
        legendTitle: {
            color: "#333333",
            fontSize: "12px",
            lineHeight: "16px",
        },
        legendSubtitle: {
            color: "#666666",
            fontSize: "9px",
            lineHeight: "12x",
            fontStyle: "italic"
        }
    });

    private static barPlotlyProps: IPlotlyProperty = {
        config: { displaylogo: false, responsive: true, modeBarButtonsToRemove: ['toggleSpikelines', 'hoverClosestCartesian', 'hoverCompareCartesian', 'zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'] },
        data: [
            {
                orientation: 'h',
                type: 'bar'
            } as any
        ],
        layout: {
            autosize: true,
            barmode: 'relative',
            colorway: ChartColors,
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
            plot_bgcolor: "#F2F2F2",
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
        const dropdownStyles: Partial<IDropdownStyles> = {
            dropdown: { width: 180,selectors: {
                ':focus .ms-Dropdown-title': {
                    color: "#333333",
                    backgroundColor: "#f3f2f1",
                },
                ':hover .ms-Dropdown-title': {
                    color: "#333333",
                    backgroundColor: "#f3f2f1",
                }
            }},
            label: { color: "#ffffff" },
            title: { color: "#ffffff", backgroundColor: "#353535", borderWidth: "0px", borderRadius: "5px", selectors: {
                ':hover': {
                    color: "#333333",
                    backgroundColor: "#f3f2f1"
                }
            }},
            caretDown: { color: "#ffffff" },
            dropdownItem: { color: "#ffffff", backgroundColor: "#353535" }
        };
        const modalStyles = {
            content : {
                top                   : '50%',
                left                  : '50%',
                right                 : 'auto',
                bottom                : 'auto',
                marginRight           : '-50%',
                paddingTop            : '5px',
                paddingRight          : '10px',
                paddingBottom         : '10px',
                transform             : 'translate(-50%, -50%)',
                fontFamily: "Segoe UI",
                color: "#FFFFFF",
                borderRadius: "5px",
                backgroundColor: "#222222",
                boxShadow: "0px 10px 15px rgba(0, 0, 0, 0.1)"
            },
            overlay: {zIndex: 1000}
        };

        const featureOptions: IDropdownOption[] = this.props.dashboardContext.modelMetadata.featureNames.map(x => { return {key: x, text: x}});

        const alternateHeight = this.props.featureBinPickerProps.featureBins[this.props.featureBinPickerProps.selectedBinIndex].labelArray.length * 60 + 106;
        const areaHeights = Math.max(300, alternateHeight);

        const accuracyKey = this.props.accuracyPickerProps.selectedAccuracyKey;
        const outcomeKey = this.props.dashboardContext.modelMetadata.predictionType === PredictionTypes.binaryClassification ? "selection_rate" : "average";
        const outcomeMetric = AccuracyOptions[outcomeKey];

        const overpredicitonKey = "overprediction";
        const underpredictionKey = "underprediction";

        const accuracyPlot = _.cloneDeep(WizardReport.barPlotlyProps);
        const opportunityPlot = _.cloneDeep(WizardReport.barPlotlyProps);
        const nameIndex = this.props.dashboardContext.groupNames.map((unuxed, i) => i);
        let howToReadAccuracySection: React.ReactNode;
        let insightsAccuracySection: React.ReactNode;
        let howToReadOutcomesSection: React.ReactNode;
        let insightsOutcomesSection: React.ReactNode;
        let accuracyChartHeader: string = "";
        let opportunityChartHeader: string = "";

        var mainChart;
        if (!this.state || !this.state.metrics) {
            this.loadData();
            mainChart = <Spinner className={WizardReport.classNames.spinner} size={SpinnerSize.large} label={localization.calculating}/>;
        }
        else {
            if (this.props.dashboardContext.modelMetadata.predictionType === PredictionTypes.binaryClassification) {
                accuracyPlot.data = [
                    {
                        x: this.state.metrics.binnedOverprediction,
                        y: nameIndex,
                        text: this.state.metrics.binnedOverprediction.map(num => this.formatNumbers((num as number), "accuracy_score", false, 2)),
                        name: localization.Metrics.overprediction,
                        width: 0.2,
                        color: ChartColors[0],
                        orientation: 'h',
                        type: 'bar',
                        textposition: 'auto',
                        hoverinfo: "skip"
                    } as any, {
                        x: this.state.metrics.binnedUnderprediction.map(x => -1 * x),
                        y: nameIndex,
                        text: this.state.metrics.binnedUnderprediction.map(num => this.formatNumbers((num as number), "accuracy_score", false, 2)),
                        name: localization.Metrics.underprediction,
                        width: 0.2,
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
                    }
                ];
                accuracyPlot.layout.xaxis.tickformat = ',.0%';
                opportunityPlot.data = [
                    {
                        x: this.state.metrics.binnedOutcome,
                        y: nameIndex,
                        text: this.state.metrics.binnedOutcome.map(num => this.formatNumbers((num as number), "selection_rate", false, 2)),
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
                    <div className={WizardReport.classNames.textRow}>{localization.Report.classificationAccuracyHowToRead3}</div>
                    <div className={WizardReport.classNames.textRow}>{localization.Report.classificationAccuracyHowToRead2}</div>
                    <div className={WizardReport.classNames.textRow}>{localization.Report.classificationAccuracyHowToRead3}</div>
                </div>);
                howToReadOutcomesSection = (<div>
                    <div className={WizardReport.classNames.textRow}>{localization.Report.classificationOutcomesHowToRead}</div>
                </div>);
            } if (this.props.dashboardContext.modelMetadata.predictionType === PredictionTypes.probability) {
                accuracyPlot.data = [
                    {
                        x: this.state.metrics.binnedOverprediction,
                        y: nameIndex,
                        text: this.state.metrics.binnedOverprediction.map(num => this.formatNumbers((num as number), "overprediction", false, 2)),
                        name: localization.Metrics.overprediction,
                        width: 0.2,
                        color: ChartColors[0],
                        orientation: 'h',
                        type: 'bar',
                        textposition: 'auto',
                        hoverinfo: "skip"
                    } as any, {
                        x: this.state.metrics.binnedUnderprediction.map(x => -1 * x),
                        y: nameIndex,
                        text: this.state.metrics.binnedUnderprediction.map(num => this.formatNumbers((num as number), "underprediction", false, 2)),
                        name: localization.Metrics.underprediction,
                        width: 0.2,
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
                const opportunityText = this.state.metrics.predictions.map(val => {
                    return localization.formatString(localization.Report.tooltipPrediction, 
                        this.formatNumbers((val as number), "average", false, 3));
                });
                opportunityPlot.data = [
                    {
                        x: this.state.metrics.predictions,
                        y: this.props.dashboardContext.binVector,
                        text: opportunityText,
                        type: 'box',
                        color: ChartColors[0],
                        boxmean: true,
                        orientation: 'h',
                        boxpoints: 'all',
                        hoverinfo: 'text',
                        hoveron: "points",
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
                    <div className={WizardReport.classNames.textRow}>{localization.Report.probabilityAccuracyHowToRead1}</div>
                    <div className={WizardReport.classNames.textRow}>{localization.Report.probabilityAccuracyHowToRead2}</div>
                    <div className={WizardReport.classNames.textRow}>{localization.Report.probabilityAccuracyHowToRead3}</div>
                </div>);
                howToReadOutcomesSection = (<div>
                    <div className={WizardReport.classNames.textRow}>{localization.Report.regressionOutcomesHowToRead}</div>
                </div>);
                opportunityChartHeader = localization.Report.distributionOfPredictions;
            } if (this.props.dashboardContext.modelMetadata.predictionType === PredictionTypes.regression) {
                const opportunityText = this.state.metrics.predictions.map(val => {
                    return localization.formatString(localization.Report.tooltipPrediction, val);
                });
                const accuracyText = this.state.metrics.predictions.map((val, index) => {
                    return `${localization.formatString(
                            localization.Report.tooltipError, 
                            this.formatNumbers((this.state.metrics.errors[index] as number), "average", false, 3))
                        }<br>${localization.formatString(
                            localization.Report.tooltipPrediction, 
                            this.formatNumbers((val as number), "average", false, 3))}`;
                });
                accuracyPlot.data = [
                    {
                        x: this.state.metrics.errors,
                        y: this.props.dashboardContext.binVector,
                        text: accuracyText,
                        type: 'box',
                        color: ChartColors[0],
                        orientation: 'h',
                        boxmean: true,
                        hoveron: "points",
                        hoverinfo: 'text',
                        boxpoints: 'all',
                        jitter: 0.4,
                        pointpos: 0,
                    } as any
                ];
                opportunityPlot.data = [
                    {
                        x: this.state.metrics.predictions,
                        y: this.props.dashboardContext.binVector,
                        text: opportunityText,
                        type: 'box',
                        color: ChartColors[0],
                        boxmean: true,
                        orientation: 'h',
                        hoveron: "points",
                        boxpoints: 'all',
                        hoverinfo: 'text',
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
                opportunityChartHeader = localization.Report.distributionOfPredictions;
                accuracyChartHeader = localization.Report.distributionOfErrors;
            }
            
            const globalAccuracyString = this.formatNumbers(this.state.metrics.globalAccuracy, accuracyKey);
            const disparityAccuracyString = this.formatNumbers(this.state.metrics.accuracyDisparity, accuracyKey);
            let selectedMetric = AccuracyOptions[this.props.accuracyPickerProps.selectedAccuracyKey];
            
            // handle custom metric case
            if (selectedMetric === undefined) {
                selectedMetric = this.props.accuracyPickerProps.accuracyOptions.find(metric => metric.key === this.props.accuracyPickerProps.selectedAccuracyKey)
            }
                        
            const globalOutcomeString = this.formatNumbers(this.state.metrics.globalOutcome, outcomeKey);
            const disparityOutcomeString = this.formatNumbers(this.state.metrics.outcomeDisparity, outcomeKey);

            const formattedBinAccuracyValues = this.state.metrics.binnedAccuracy.map(value => 
                this.formatNumbers(value, accuracyKey));
            const formattedBinOutcomeValues = this.state.metrics.binnedOutcome.map(value => 
                this.formatNumbers(value, outcomeKey));
            const formattedBinOverPredictionValues = this.state.metrics.binnedOverprediction.map(value => 
                this.formatNumbers(value, overpredicitonKey));
            const formattedBinUnderPredictionValues = this.state.metrics.binnedUnderprediction.map(value => 
                this.formatNumbers(value, underpredictionKey));

            const globalOverpredictionString = this.formatNumbers(this.state.metrics.globalOverprediction, outcomeKey);
            const globalUnderpredictionString = this.formatNumbers(this.state.metrics.globalUnderprediction, outcomeKey);

            const overallMetrics = [globalAccuracyString, globalOutcomeString, globalOverpredictionString, globalUnderpredictionString];
            const formattedBinValues = [formattedBinAccuracyValues, formattedBinOutcomeValues, formattedBinOverPredictionValues, formattedBinUnderPredictionValues];
            const metricLabels = [AccuracyOptions[accuracyKey].title, AccuracyOptions[outcomeKey].title, AccuracyOptions[overpredicitonKey].title, AccuracyOptions[underpredictionKey].title];

            mainChart = 
                    <div className={WizardReport.classNames.main}>
                        <div className={WizardReport.classNames.mainLeft}>
                            <div className={WizardReport.classNames.overallArea} style={{height: !this.state.expandAttributes && "150px" || this.state.expandAttributes && `${150 + 50*(areaHeights/150)}px` }}>
                                <OverallTable
                                    binGroup={this.props.dashboardContext.modelMetadata.featureNames[this.props.featureBinPickerProps.selectedBinIndex]}
                                    binLabels={this.props.dashboardContext.groupNames}
                                    formattedBinValues={formattedBinValues}
                                    metricLabels={metricLabels}
                                    overallMetrics={overallMetrics}
                                    expandAttributes={this.state.expandAttributes}
                                    binValues={this.state.metrics.binnedAccuracy}/>
                            </div>
                            <div className={WizardReport.classNames.expandAttributes} onClick={this.expandAttributes}>
                                <svg style={{verticalAlign: "middle", marginRight: "10px"}} width="9" height="6" viewBox="0 0 9 6" fill="none" xmlns="http://www.w3.org/2000/svg">
                                    <path d={this.state.expandAttributes && "M8 5L4 1L0 5" || !this.state.expandAttributes && "M0 1L4 5L8 1"} stroke="#666666" stroke-linecap="round" stroke-linejoin="round"/>
                                </svg>
                                <span>{this.state.expandAttributes && localization.Report.collapseSensitiveAttributes || !this.state.expandAttributes && localization.Report.expandSensitiveAttributes}</span>
                            </div>
                            <div className={WizardReport.classNames.equalizedOdds}>{localization.Report.equalizedOddsDisparity}</div>
                            <div className={WizardReport.classNames.howTo}>
                                    <ActionButton onClick={this.handleOpenModalHelp}><div className={WizardReport.classNames.infoButton}>i</div>{localization.ModelComparison.howToRead}</ActionButton>
                                    <ReactModal
                                        style={modalStyles}
                                        appElement={document.getElementById('app') as HTMLElement}
                                        isOpen={this.state.showModalHelp}
                                        contentLabel="Minimal Modal Example"
                                        >
                                        {/* <ActionButton className={WizardReport.classNames.closeButton} onClick={this.handleCloseModalHelp}>x</ActionButton> */}
                                        <p className={WizardReport.classNames.modalContentHelp}>{localization.Report.classificationAccuracyHowToRead1}<br/><br/>{localization.Report.classificationAccuracyHowToRead2}<br/><br/>{localization.Report.classificationAccuracyHowToRead3}<br /><br /><ActionButton className={WizardReport.classNames.doneButton} onClick={this.handleCloseModalHelp}>Done</ActionButton></p>
                                    </ReactModal>
                            </div>
                            <div className={WizardReport.classNames.presentationArea} style={{height: `${areaHeights}px`}}>
                                <SummaryTable 
                                    binGroup={this.props.dashboardContext.modelMetadata.featureNames[this.props.featureBinPickerProps.selectedBinIndex]}
                                    binLabels={this.props.dashboardContext.groupNames}
                                    formattedBinValues={formattedBinAccuracyValues}
                                    metricLabel={selectedMetric.title}
                                    binValues={this.state.metrics.binnedAccuracy}/>
                                <div className={WizardReport.classNames.chartWrapper}>
                                    <div className={WizardReport.classNames.chartHeader}>{accuracyChartHeader}</div>
                                    <div className={WizardReport.classNames.chartBody}>
                                        <AccessibleChart
                                            plotlyProps={accuracyPlot}
                                            sharedSelectionContext={undefined}
                                            theme={undefined}
                                        />
                                    </div>
                                </div>
                            </div>
                            <div className={WizardReport.classNames.textRow}>
                                <div className={WizardReport.classNames.colorBlock} style={{backgroundColor: ChartColors[1]}}/>
                                <div>
                                    <div className={WizardReport.classNames.legendTitle}>{localization.Report.underestimationError}</div>
                                    <div className={WizardReport.classNames.legendSubtitle}>{localization.Report.underpredictionExplanation}</div>
                                </div>
                            </div>
                            <div className={WizardReport.classNames.textRow}>
                                <div className={WizardReport.classNames.colorBlock} style={{backgroundColor: ChartColors[0]}}/>
                                <div>
                                    <div className={WizardReport.classNames.legendTitle}>{localization.Report.overestimationError}</div>
                                    <div className={WizardReport.classNames.legendSubtitle}>{localization.Report.overpredictionExplanation}</div>
                                </div>
                            </div>
                        </div>
                        <div className={WizardReport.classNames.mainRight}>
                            <div className={WizardReport.classNames.insights}>
                                <svg style={{verticalAlign: "middle", marginRight: "10px"}} width="24" height="28" viewBox="0 0 24 28" fill="none" xmlns="http://www.w3.org/2000/svg">
                                    <path fill-rule="evenodd" clip-rule="evenodd" d="M11.5812 4.30233C15.9998 4.06977 19.6626 7.61628 19.6626 11.9767C19.6626 13.7791 19.0231 15.4651 17.9184 16.8023C16.5812 18.4884 15.8835 20.3488 15.8835 22.2674V22.6744C15.8835 23.0233 15.651 23.2558 15.3021 23.2558H8.49981C8.15097 23.2558 7.91842 23.0233 7.91842 22.6744V22.3256C7.91842 20.2907 7.27888 18.3721 6.05795 16.9767C4.77888 15.5233 4.13935 13.6047 4.19749 11.6279C4.3719 7.73256 7.62772 4.47674 11.5812 4.30233ZM9.13935 22.093H14.7789C14.7789 20 15.5928 17.907 17.0463 16.1047C17.9766 14.8837 18.4998 13.4884 18.4998 11.9767C18.4998 8.25581 15.4184 5.23256 11.6393 5.40698C8.32539 5.5814 5.59283 8.31395 5.41842 11.686C5.36028 13.314 5.88353 14.9419 6.98818 16.2209C8.32539 17.7907 9.08121 19.8837 9.13935 22.093Z" fill="#666666"/>
                                    <path d="M15.3025 24.593H8.73276C8.38393 24.593 8.15137 24.8256 8.15137 25.1744C8.15137 25.5233 8.38393 25.7558 8.73276 25.7558H15.3025C15.6514 25.7558 15.8839 25.5233 15.8839 25.1744C15.8839 24.8256 15.6514 24.593 15.3025 24.593Z" fill="#666666"/>
                                    <path d="M14.1395 26.7442H9.89536C9.54652 26.7442 9.31396 26.9767 9.31396 27.3256C9.31396 27.6744 9.54652 27.907 9.89536 27.907H14.1395C14.4884 27.907 14.7209 27.6744 14.7209 27.3256C14.7209 26.9767 14.4884 26.7442 14.1395 26.7442Z" fill="#666666"/>
                                    <path d="M11.93 3.37209C12.2789 3.37209 12.5114 3.13953 12.5114 2.7907V0.581395C12.5114 0.232558 12.2789 0 11.93 0C11.5812 0 11.3486 0.232558 11.3486 0.581395V2.7907C11.3486 3.13953 11.5812 3.37209 11.93 3.37209Z" fill="#666666"/>
                                    <path d="M18.3255 6.10465C18.5 6.10465 18.6162 6.04651 18.7325 5.93023L20.2441 4.36047C20.4767 4.12791 20.4767 3.77907 20.2441 3.54651C20.0116 3.31395 19.6627 3.31395 19.4302 3.54651L17.9186 5.11628C17.686 5.34884 17.686 5.69767 17.9186 5.93023C18.0348 6.04651 18.2093 6.10465 18.3255 6.10465Z" fill="#666666"/>
                                    <path d="M5.12794 18.3139L3.61631 19.8837C3.38375 20.1163 3.38375 20.4651 3.61631 20.6977C3.73259 20.8139 3.84887 20.8721 4.02329 20.8721C4.19771 20.8721 4.31399 20.8139 4.43027 20.6977L5.94189 19.1279C6.17445 18.8953 6.17445 18.5465 5.94189 18.3139C5.70934 18.0233 5.3605 18.0814 5.12794 18.3139Z" fill="#666666"/>
                                    <path d="M23.151 11.5116H20.9999C20.651 11.5116 20.4185 11.7442 20.4185 12.093C20.4185 12.4419 20.651 12.6744 20.9999 12.6744H23.151C23.4999 12.6744 23.7324 12.4419 23.7324 12.093C23.7324 11.7442 23.4417 11.5116 23.151 11.5116Z" fill="#666666"/>
                                    <path d="M2.86049 11.5116H0.709325C0.360488 11.5116 0.12793 11.7442 0.12793 12.093C0.12793 12.4419 0.360488 12.6744 0.709325 12.6744H2.86049C3.20933 12.6744 3.44188 12.4419 3.44188 12.093C3.44188 11.7442 3.20933 11.5116 2.86049 11.5116Z" fill="#666666"/>
                                    <path d="M18.7325 18.3139C18.5 18.0814 18.1511 18.0814 17.9186 18.3139C17.686 18.5465 17.686 18.8953 17.9186 19.1279L19.4302 20.6977C19.5465 20.8139 19.7209 20.8721 19.8372 20.8721C20.0116 20.8721 20.1279 20.8139 20.2441 20.6977C20.4767 20.4651 20.4767 20.1163 20.2441 19.8837L18.7325 18.3139Z" fill="#666666"/>
                                    <path d="M5.12794 5.93023C5.24422 6.04651 5.41864 6.10465 5.53492 6.10465C5.70934 6.10465 5.82562 6.04651 5.94189 5.93023C6.17445 5.69767 6.17445 5.34884 5.94189 5.11628L4.43027 3.54651C4.19771 3.31395 3.84887 3.31395 3.61631 3.54651C3.38375 3.77907 3.38375 4.12791 3.61631 4.36047L5.12794 5.93023Z" fill="#666666"/>
                                </svg>
                                <span style={{verticalAlign: "middle"}}>{localization.ModelComparison.insights}</span>
                            </div>
                            <div className={WizardReport.classNames.insightsText}>{localization.loremIpsum}</div>
                            <div className={WizardReport.classNames.downloadReport}>
                                <svg style={{verticalAlign: "middle", marginRight: "10px"}} width="17" height="18" viewBox="0 0 17 18" fill="none" xmlns="http://www.w3.org/2000/svg">
                                    <path d="M16.4453 16.972C16.4453 17.3459 16.142 17.6492 15.7682 17.6492L1.47353 17.6492C1.0997 17.6492 0.79641 17.3459 0.79641 16.972C0.79641 16.5982 1.0997 16.2949 1.47353 16.2949H15.7682C15.9478 16.2949 16.1112 16.3649 16.2322 16.4789C16.3633 16.6023 16.4453 16.7775 16.4453 16.972Z" fill="#5A53FF"/>
                                    <path d="M9.57503 11.717L9.57503 0.877332C9.57503 0.499978 9.27728 0.194336 8.90935 0.194336C8.54172 0.194336 8.24397 0.499978 8.24397 0.877332L8.24397 11.717L4.92809 8.34501C4.66561 8.09107 4.25272 8.09811 3.99879 8.36204C3.74455 8.62476 3.74136 9.0474 3.99144 9.31482L8.44205 13.8395C8.70158 14.1034 9.11947 14.1034 9.3787 13.8395L13.8293 9.31482C13.9072 9.23958 13.9654 9.14731 14.0004 9.04566C14.0239 8.97686 14.0368 8.90397 14.038 8.82994C14.0412 8.64534 13.9716 8.46785 13.8455 8.33736C13.7194 8.20627 13.5466 8.13399 13.367 8.13519C13.28 8.13573 13.1951 8.15397 13.1163 8.18749C13.0329 8.22331 12.9565 8.27681 12.8927 8.34501L9.57503 11.717Z" fill="#5A53FF"/>
                                </svg>
                                <span style={{verticalAlign: "middle"}}>{localization.ModelComparison.downloadReport}</span>
                            </div>
                        </div>
                    </div>
        }

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
                    </div>}
                    <div className={WizardReport.classNames.modelLabel}>
                        {localization.Report.assessmentResults} <b>{this.props.dashboardContext.modelNames[this.props.selectedModelIndex]}</b>
                    </div>
                    <div className={WizardReport.classNames.headerOptions}>
                        <Dropdown
                            className={WizardReport.classNames.dropDown}
                            // label="Feature"
                            defaultSelectedKey={this.props.dashboardContext.modelMetadata.featureNames[this.props.featureBinPickerProps.selectedBinIndex]}
                            options={featureOptions}
                            disabled={false}
                            onChange={this.featureChanged}
                            styles={dropdownStyles}
                        />
                    </div>
            </div>
            {mainChart}
        </div>);
    }

    private readonly formatNumbers = (value: number, key: string, isRatio: boolean = false, sigDigits: number = 3): string => {
        if (value === null || value === undefined || value === NaN) {
            return NaN.toString();
        }
        const styleObject = {maximumSignificantDigits: sigDigits};
        if (AccuracyOptions[key] && AccuracyOptions[key].isPercentage && !isRatio) {
            (styleObject as any).style = "percent";
        }
        return value.toLocaleString(undefined, styleObject);
    }

    private readonly clearModelSelection = (): void => {
        this.props.selections.onSelect([]);
    }

    private readonly expandAttributes = (): void => {
        this.setState({ expandAttributes: !this.state.expandAttributes });
    }

    private readonly onEditConfigs = (): void => {
        if (this.props.modelCount > 1) {
            this.props.selections.onSelect([]);
        }
        this.props.onEditConfigs();
    }

    private readonly handleOpenModalHelp = (event): void => {
        this.setState({ showModalHelp: true });
    }

    private readonly handleCloseModalHelp = (event): void => {
        this.setState({ showModalHelp: false });
    }

    private readonly featureChanged = (ev: React.FormEvent<HTMLInputElement>, option: IDropdownOption): void => {
        const featureKey = option.key.toString();
        if (this.state.featureKey !== featureKey) {
            this.props.featureBinPickerProps.selectedBinIndex = this.props.dashboardContext.modelMetadata.featureNames.indexOf(featureKey);
            this.setState({featureKey: featureKey, metrics: undefined});
        }
    }

    private async loadData(): Promise<void> {
        try {
            let binnedFNR: number[];
            let binnedFPR: number[];
            let overallOverprediction: number;
            let overallUnderprediction: number;
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
                overallUnderprediction = (await this.props.metricsCache.getMetric(
                        this.props.dashboardContext.binVector,
                        this.props.featureBinPickerProps.selectedBinIndex, 
                        this.props.selectedModelIndex,
                        "underprediction")).global;
                binnedOverprediction = (await this.props.metricsCache.getMetric(
                    this.props.dashboardContext.binVector,
                    this.props.featureBinPickerProps.selectedBinIndex, 
                    this.props.selectedModelIndex,
                    "overprediction")).bins;
                overallOverprediction = (await this.props.metricsCache.getMetric(
                    this.props.dashboardContext.binVector,
                    this.props.featureBinPickerProps.selectedBinIndex, 
                    this.props.selectedModelIndex,
                    "overprediction")).global;    
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
                    accuracyDisparity: accuracyDisparity,
                    globalOutcome: outcomes.global,
                    binnedOutcome: outcomes.bins,
                    outcomeDisparity: outcomeDisparity,
                    predictions: predictions,
                    errors: errors,
                    globalOverprediction: overallOverprediction,
                    globalUnderprediction: overallUnderprediction,
                    binnedOverprediction: binnedOverprediction,
                    binnedUnderprediction: binnedUnderprediction
                }
            });
        } catch {
            // todo;
        }
    }
}
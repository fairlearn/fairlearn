import React from "react";
import ReactModal from "react-modal";
import { AccessibleChart, ChartBuilder, IPlotlyProperty, PlotlyMode, SelectionContext } from "mlchartlib";
import { IFairnessContext } from "../IFairnessContext";
import _ from "lodash";
import { MetricsCache } from "../MetricsCache";
import { IAccuracyPickerProps, IParityPickerProps, IFeatureBinPickerProps } from "../FairnessWizard";
import { ParityModes, ParityOptions } from "../ParityMetrics";
import { Stack, StackItem } from "office-ui-fabric-react/lib/Stack";
import { ChoiceGroup, IChoiceGroupOption } from 'office-ui-fabric-react/lib/ChoiceGroup';
import { Dropdown, DropdownMenuItemType, IDropdownStyles, IDropdownOption } from 'office-ui-fabric-react/lib/Dropdown';
import { localization } from "../Localization/localization";
import { Spinner, SpinnerSize } from "office-ui-fabric-react/lib/Spinner";
import { mergeStyleSets } from "@uifabric/styling";
import { ActionButton } from "office-ui-fabric-react/lib/Button";
import { AccuracyOptions } from "../AccuracyMetrics";
import { FormatMetrics } from "../FormatMetrics";
import { PredictionTypes } from "../IFairnessProps";

export interface IModelComparisonProps {
    showIntro: boolean,
    dashboardContext: IFairnessContext;
    selections: SelectionContext;
    metricsCache: MetricsCache;
    modelCount: number;
    accuracyPickerProps: IAccuracyPickerProps;
    parityPickerProps: IParityPickerProps;
    featureBinPickerProps: IFeatureBinPickerProps;
    onHideIntro: () => void;
    onEditConfigs: () => void;
}

export interface IState {
    showModalIntro?: boolean;
    showModalHelp?: boolean;
    featureKey?: string;
    accuracyKey?: string;
    parityKey?: string;
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
                mode: PlotlyMode.textMarkers,
                marker: {
                    size: 14
                },
                textposition: 'top',
                type: 'scatter',
                xAccessor: 'Accuracy',
                yAccessor: 'Parity',
                hoverinfo: 'text'
            } as any
        ],
        layout: {
            autosize: true,
            plot_bgcolor:"#F2F2F2",
            font: {
                size: 10
            },
            margin: {
                t: 4,
                r:0
            },
            hovermode: 'closest',
            xaxis: {
                showgrid: false,
                automargin: true,
                fixedrange: true,
                mirror: true,
                title:{
                    text: 'Error'
                }
            },
            yaxis: {
                showgrid: false,
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
            backgroundColor: "#222222",
            padding: "0 50px",
            height: "90px",
            display: "inline-flex",
            flexDirection: "row",
            justifyContent: "space-between",
            alignItems: "center"
        },
        headerTitle: {
            color: "#FFFFFF",
            fontSize: "30px",
            lineHeight: "43px",
            fontWeight: "300"
        },
        headerOptions: {
            backgroundColor: "#222222",
            padding: "0 40px"
        },
        dropDown: {
            margin: "10px 10px",
            display: "inline-block"
        },
        infoButton: {
            float: "left",
            width: "15px",
            height: "15px",
            textAlign: "center",
            fontFamily: "Arial",
            fontSize: "12px",
            lineHeight: "15px",
            fontWeight: "400",
            borderRadius: "50%",
            border: "1px solid",
            marginTop: "3px",
            marginRight: "3px"
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
        modalContentIntro: {
            float: 'left',
            textAlign: 'center',
            paddingTop: '10px',
            paddingRight: '20px'
        },
        modalContentHelp: {
            float: 'left',
            textAlign: 'center',
            paddingTop: '10px',
            paddingRight: '20px',
        },
        editButton: {
            color: "#333333",
            fontSize: "12px",
            lineHeight: "20px",
            fontWeight: "400"
        },
        howTo: {
            paddingTop: "20px",
            paddingLeft: "100px"
        },
        main: {
            display: "flex",
            flexDirection: "row",
        },
        mainLeft: {
            width: "75%",
        },
        mainRight: {
            width: "240px",
            marginLeft: "20px",
            padding: "30px 0 0 35px",
            backgroundColor: "#f2f2f2",
            boxShadow: "-1px 0px 0px #D2D2D2"
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
            paddingRight: "15px",
        },
        downloadReport: {
            color: "#333333",
            fontSize: "12px",
            lineHeight: "16px",
            fontWeight: "normal",
            paddingTop: "20px",
            paddingBottom: "20px",
            paddingLeft: "30px",
        },
        chart: {
            padding: "0px 0 0 0",
            flex: 1
        },
        textSection: {
            padding: "20px 0",
            borderBottom: "1px solid #cccc"
        },
        textSectionLast: {
            padding: "20px 0",
        },
        radio: {
            paddingBottom: "30px",
            paddingLeft: "75px"
        }
    });

    constructor(props: IModelComparisonProps) {
        super(props);
        this.state = {
            disparityInOutcomes: true,
            showModalIntro: this.props.showIntro,
            accuracyKey: this.props.accuracyPickerProps.selectedAccuracyKey
        };
    }

    public render(): React.ReactNode {
        const featureOptions: IDropdownOption[] = this.props.dashboardContext.modelMetadata.featureNames.map(x => { return {key: x, text: x}});
        const accuracyOptions: IDropdownOption[] = this.props.accuracyPickerProps.accuracyOptions.map(x => { return {key: x.key, text: x.title}});
        const parityOptions: IDropdownOption[] = this.props.parityPickerProps.parityOptions.map(x => { return {key: x.key, text: x.title}});
         
        const dropdownStyles: Partial<IDropdownStyles> = {
            label: { color: "#ffffff" },
            dropdown: { width: 180, selectors: {
                ':focus .ms-Dropdown-title': {
                    color: "#333333",
                    backgroundColor: "#f3f2f1",
                },
                ':hover .ms-Dropdown-title': {
                    color: "#333333",
                    backgroundColor: "#f3f2f1",
                },
            }},
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
         
        var mainChart;
        if (!this.state || this.state.accuracyArray === undefined || this.state.disparityArray === undefined) {
            this.loadData();
            mainChart = <Spinner className={ModelComparisonChart.classNames.spinner} size={SpinnerSize.large} label={localization.calculating}/>;
        }
        else {
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
           
            let selectedMetric = AccuracyOptions[this.props.accuracyPickerProps.selectedAccuracyKey];
            
             // handle custom metric case
            if (selectedMetric === undefined) {
                selectedMetric = this.props.accuracyPickerProps.accuracyOptions.find(metric => metric.key === this.props.accuracyPickerProps.selectedAccuracyKey)
            }


            // const insights2 = [selectedMetric.title,
            //     localization.ModelComparison.rangesFrom,
            //     <strong>{formattedMinAccuracy}</strong>,
            //     localization.ModelComparison.to,
            //     <strong>{formattedMaxAccuracy}</strong>,
            //     localization.ModelComparison.period,
            //     localization.ModelComparison.disparity,
            //     localization.ModelComparison.rangesFrom,
            //     <strong>{formattedMinDisparity}</strong>,
            //     localization.ModelComparison.to,
            //     <strong>{formattedMaxDisparity}</strong>,
            //     localization.ModelComparison.period
            // ];
    
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
                series.text = this.props.dashboardContext.modelNames;
                return series;
            });
            
            const accuracyMetricTitle = selectedMetric.title 
            const parityMetricTitle = ParityOptions[this.props.parityPickerProps.selectedParityKey].title;
            props.layout.xaxis.title = accuracyMetricTitle;
            props.layout.yaxis.title = parityMetricTitle;
            
            mainChart = <div className={ModelComparisonChart.classNames.main}>
                            <div className={ModelComparisonChart.classNames.mainLeft}>
                                <div className={ModelComparisonChart.classNames.howTo}>
                                    <ReactModal
                                        style={modalStyles}
                                        appElement={document.getElementById('app') as HTMLElement}
                                        isOpen={this.state.showModalIntro}
                                        contentLabel="Intro Modal Example"
                                        >
                                        {/* <ActionButton className={ModelComparisonChart.classNames.closeButton} onClick={this.handleCloseModalIntro}>x</ActionButton> */}
                                        <p className={ModelComparisonChart.classNames.modalContentIntro}>Each model is a selectable point. <br />Click or tap on model for it's<br />full fairness assessment. <br /><br /><ActionButton className={ModelComparisonChart.classNames.doneButton} onClick={this.handleCloseModalIntro}>Done</ActionButton></p>
                                    </ReactModal>
                                    <ActionButton onClick={this.handleOpenModalHelp}><div className={ModelComparisonChart.classNames.infoButton}>i</div>{localization.ModelComparison.howToRead}</ActionButton>
                                    <ReactModal
                                        style={modalStyles}
                                        appElement={document.getElementById('app') as HTMLElement}
                                        isOpen={this.state.showModalHelp}
                                        contentLabel="Minimal Modal Example"
                                        >
                                        {/* <ActionButton className={ModelComparisonChart.classNames.closeButton} onClick={this.handleCloseModalHelp}>x</ActionButton> */}
                                        <p className={ModelComparisonChart.classNames.modalContentHelp}>The <b>x-axis</b> represents accuracy, <br />with higher being better.<br /><br />The <b>y-axis</b> represents disparity, <br /> with lower being better.<br /><br /><ActionButton className={ModelComparisonChart.classNames.doneButton} onClick={this.handleCloseModalHelp}>Done</ActionButton></p>
                                    </ReactModal>
                                </div>
                                <div className={ModelComparisonChart.classNames.chart}>
                                    <AccessibleChart
                                        plotlyProps={props}
                                        sharedSelectionContext={this.props.selections}
                                        theme={undefined}
                                    />
                                </div>
                            </div>
                            <div className={ModelComparisonChart.classNames.mainRight}>
                                <div className={ModelComparisonChart.classNames.insights}>{localization.ModelComparison.insights}</div>
                                <div className={ModelComparisonChart.classNames.insightsText}>
                                    <div className={ModelComparisonChart.classNames.textSection}>{insights2}</div>
                                    <div className={ModelComparisonChart.classNames.textSection}>{insights3}</div>
                                    <div className={ModelComparisonChart.classNames.textSectionLast}>{insights4}</div>
                                </div>
                                <div className={ModelComparisonChart.classNames.downloadReport}>{localization.ModelComparison.downloadReport}</div>
                            </div>
                        </div>;
        }

        return (
            <Stack className={ModelComparisonChart.classNames.frame}>
                <div className={ModelComparisonChart.classNames.header}>
                    <h2 className={ModelComparisonChart.classNames.headerTitle}>{localization.ModelComparison.title} <b>assessment</b></h2>
                    {/* <ActionButton iconProps={{iconName: "Edit"}} onClick={this.props.onEditConfigs} className={ModelComparisonChart.classNames.editButton}>{localization.Report.editConfiguration}</ActionButton> */}
                </div>
                <div className={ModelComparisonChart.classNames.headerOptions}>
                    <Dropdown
                        className={ModelComparisonChart.classNames.dropDown}
                        // label="Feature"
                        defaultSelectedKey={this.props.dashboardContext.modelMetadata.featureNames[this.props.featureBinPickerProps.selectedBinIndex]}
                        options={featureOptions}
                        disabled={false}
                        onChange={this.featureChanged}
                        styles={dropdownStyles}
                    />
                    <Dropdown
                        className={ModelComparisonChart.classNames.dropDown}
                        // label="Accuracy"
                        defaultSelectedKey={this.props.accuracyPickerProps.selectedAccuracyKey}
                        options={accuracyOptions}
                        disabled={false}
                        onChange={this.accuracyChanged}
                        styles={dropdownStyles}
                    />
                    <Dropdown
                        className={ModelComparisonChart.classNames.dropDown}
                        // label="Parity"
                        defaultSelectedKey={this.props.parityPickerProps.selectedParityKey}
                        options={parityOptions}
                        disabled={false}
                        onChange={this.parityChanged}
                        styles={dropdownStyles}
                    />
                </div>
                {mainChart}
                {/* <div>
                    <ChoiceGroup
                        className={ModelComparisonChart.classNames.radio}
                        selectedKey={this.state.disparityInOutcomes ? "outcomes" : "accuracy"}
                        options={[
                            {
                            key: 'accuracy',
                            text: localization.formatString(localization.ModelComparison.disparityInAccuracy, 
                                accuracyMetricTitle.toLowerCase()) as string
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
                </div> */}
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
                this.props.parityPickerProps.selectedParityKey : "average") :
                this.props.accuracyPickerProps.selectedAccuracyKey;
            const parityMode = this.props.parityPickerProps.parityOptions.filter(option => option.key == this.props.parityPickerProps.selectedParityKey)[0].parityModes[0];
            const disparityPromises = new Array(this.props.modelCount).fill(0).map((unused, modelIndex) => {
                return this.props.metricsCache.getDisparityMetric(
                    this.props.dashboardContext.binVector,
                    this.props.featureBinPickerProps.selectedBinIndex,
                    modelIndex, 
                    disparityMetric,
                    parityMode);
            });

            const accuracyArray = (await Promise.all(accuracyPromises)).map(metric => metric.global);
            const disparityArray = await Promise.all(disparityPromises);
            this.setState({accuracyArray, disparityArray});
        } catch {
            // todo;
        }
    }

    private readonly featureChanged = (ev: React.FormEvent<HTMLInputElement>, option: IDropdownOption): void => {
        const featureKey = option.key.toString();
        if (this.state.featureKey !== featureKey) {
            this.props.featureBinPickerProps.selectedBinIndex = this.props.dashboardContext.modelMetadata.featureNames.indexOf(featureKey);
            this.setState({featureKey: featureKey, disparityArray: undefined});
        }
    }

    private readonly accuracyChanged = (ev: React.FormEvent<HTMLInputElement>, option: IDropdownOption): void => {
        const accuracyKey = option.key.toString();
        if (this.state.accuracyKey !== accuracyKey) {
            this.props.accuracyPickerProps.selectedAccuracyKey = accuracyKey;
            this.setState({accuracyKey: accuracyKey, disparityArray: undefined});
        }
    }

    private readonly parityChanged = (ev: React.FormEvent<HTMLInputElement>, option: IDropdownOption): void => {
        const parityKey = option.key.toString();
        if (this.state.parityKey !== parityKey) {
            this.props.parityPickerProps.selectedParityKey = parityKey;
            this.setState({parityKey: parityKey, disparityArray: undefined});
        }
    }

    private readonly disparityChanged = (ev: React.FormEvent<HTMLInputElement>, option: IChoiceGroupOption): void => {
        const disparityInOutcomes = option.key !== "accuracy";
        if (this.state.disparityInOutcomes !== disparityInOutcomes) {
            this.setState({disparityInOutcomes, disparityArray: undefined});
        }
    }

    private readonly handleCloseModalIntro = (event): void => {
        this.setState({ showModalIntro: false });
        this.props.onHideIntro();
    }

    private readonly handleOpenModalHelp = (event): void => {
        this.setState({ showModalHelp: true });
    }

    private readonly handleCloseModalHelp = (event): void => {
        this.setState({ showModalHelp: false });
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
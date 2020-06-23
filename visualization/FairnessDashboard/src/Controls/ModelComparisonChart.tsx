import React from "react";
import ReactModal from "react-modal";
import { AccessibleChart, ChartBuilder, IPlotlyProperty, PlotlyMode, SelectionContext } from "mlchartlib";
import { IFairnessContext } from "../IFairnessContext";
import _ from "lodash";
import { MetricsCache } from "../MetricsCache";
import { IAccuracyPickerProps, IParityPickerProps, IFeatureBinPickerProps } from "../FairnessWizard";
import { ParityOptions } from "../ParityMetrics";
import { Stack } from "office-ui-fabric-react/lib/Stack";
import { Dropdown, IDropdownStyles, IDropdownOption } from 'office-ui-fabric-react/lib/Dropdown';
import { getTheme, Text, Icon } from "office-ui-fabric-react";
import { ActionButton } from "office-ui-fabric-react/lib/Button";
import { Spinner, SpinnerSize } from "office-ui-fabric-react/lib/Spinner";
import { AccuracyOptions } from "../AccuracyMetrics";
import { FormatMetrics } from "../FormatMetrics";
import { localization } from "../Localization/localization";
import { ModelComparisionChartStyles } from "./ModelComparisionChart.styles";
import { PredictionTypes } from "../IFairnessProps";

const theme = getTheme();
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
            plot_bgcolor: theme.semanticColors.bodyFrameBackground,
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
                linecolor: theme.semanticColors.disabledBorder,
                linewidth: 1,
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

    constructor(props: IModelComparisonProps) {
        super(props);
        this.state = {
            showModalIntro: this.props.showIntro,
            accuracyKey: this.props.accuracyPickerProps.selectedAccuracyKey,
            parityKey: this.props.parityPickerProps.selectedParityKey
        };
    }

    public render(): React.ReactNode {
        const featureOptions: IDropdownOption[] = this.props.dashboardContext.modelMetadata.featureNames.map(x => { return {key: x, text: x}});
        const accuracyOptions: IDropdownOption[] = this.props.accuracyPickerProps.accuracyOptions.map(x => { return {key: x.key, text: x.title}});
        const parityOptions: IDropdownOption[] = this.props.parityPickerProps.parityOptions.map(x => { return {key: x.key, text: x.title}});
         
        const dropdownStyles: Partial<IDropdownStyles> = {
            dropdown: { width: 180 },
            title: { borderRadius: "5px" }
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

        const styles = ModelComparisionChartStyles();
        
        var mainChart;
        if (!this.state || this.state.accuracyArray === undefined || this.state.disparityArray === undefined) {
            this.loadData();
            mainChart = <Spinner className={styles.spinner} size={SpinnerSize.large} label={localization.calculating}/>;
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
            
            const InsightsIcon = () => <Icon iconName="CRMCustomerInsightsApp" className={styles.insightsIcon} />;
            const DownloadIcon = () => <Icon iconName="Download" className={styles.downloadIcon} />;

            mainChart = <div className={styles.main}>
                            <div className={styles.mainLeft}>
                                <div className={styles.howTo}>
                                    <ReactModal
                                        style={modalStyles}
                                        appElement={document.getElementById('app') as HTMLElement}
                                        isOpen={this.state.showModalIntro}
                                        contentLabel="Intro Modal Example"
                                        >
                                        {/* <ActionButton className={ModelComparisonChart.classNames.closeButton} onClick={this.handleCloseModalIntro}>x</ActionButton> */}
                                        <p className={styles.modalContentIntro}>Each model is a selectable point. <br />Click or tap on model for it's<br />full fairness assessment. <br /><br /><ActionButton className={styles.doneButton} onClick={this.handleCloseModalIntro}>Done</ActionButton></p>
                                    </ReactModal>
                                    <ActionButton onClick={this.handleOpenModalHelp}><div className={styles.infoButton}>i</div>{localization.ModelComparison.howToRead}</ActionButton>
                                    <ReactModal
                                        style={modalStyles}
                                        appElement={document.getElementById('app') as HTMLElement}
                                        isOpen={this.state.showModalHelp}
                                        contentLabel="Minimal Modal Example"
                                        >
                                        {/* <ActionButton className={ModelComparisonChart.classNames.closeButton} onClick={this.handleCloseModalHelp}>x</ActionButton> */}
                                        <p className={styles.modalContentHelp}>The <b>x-axis</b> represents accuracy, <br />with higher being better.<br /><br />The <b>y-axis</b> represents disparity, <br /> with lower being better.<br /><br /><ActionButton className={styles.doneButton} onClick={this.handleCloseModalHelp}>Done</ActionButton></p>
                                    </ReactModal>
                                </div>
                                <div className={styles.chart}>
                                    <AccessibleChart
                                        plotlyProps={props}
                                        sharedSelectionContext={this.props.selections}
                                        theme={undefined}
                                    />
                                </div>
                            </div>
                            <div className={styles.mainRight}>
                                <div className={styles.insights}>
                                    <InsightsIcon />
                                    <Text className={styles.insights} block>{localization.ModelComparison.insights}</Text>   
                                </div>
                                <div className={styles.insightsText}>
                                    <Text className={styles.textSection} block>{insights2}</Text>
                                    <Text className={styles.textSection} block>{insights3}</Text>
                                    <Text className={styles.textSection} block>{insights4}</Text>
                                </div>                                
                                <div className={styles.downloadReport}>
                                    <DownloadIcon />
                                    <Text style={{verticalAlign: "middle"}}>{localization.ModelComparison.downloadReport}</Text>
                                </div>                                
                            </div>
                        </div>;
        }

        return (
            <Stack className={styles.frame}>
                <div className={styles.header}>
                    <Text variant={"large"} className={styles.headerTitle} block>{localization.ModelComparison.title} <b>assessment</b></Text>
                    {/*<ActionButton iconProps={{iconName: "Edit"}} onClick={this.props.onEditConfigs} className={styles.editButton}>{localization.Report.editConfiguration}</ActionButton>*/}
                </div>
                <div className={styles.headerOptions}>
                    <Dropdown
                        className={styles.dropDown}
                        defaultSelectedKey={this.props.dashboardContext.modelMetadata.featureNames[this.props.featureBinPickerProps.selectedBinIndex]}
                        options={featureOptions}
                        disabled={false}
                        onChange={this.featureChanged}
                        styles={dropdownStyles}
                    />
                    <Dropdown
                        className={styles.dropDown}
                        defaultSelectedKey={this.props.accuracyPickerProps.selectedAccuracyKey}
                        options={accuracyOptions}
                        disabled={false}
                        onChange={this.accuracyChanged}
                        styles={dropdownStyles}
                    />
                    <Dropdown
                        className={styles.dropDown}
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
                        className={styles.radio}
                        selectedKey={this.state.disparityInOutcomes ? "outcomes" : "accuracy"}
                        options={[
                            {
                            key: 'accuracy',
                            text: localization.formatString(localization.ModelComparison.disparityInAccuracy, metricTitleAppropriateCase) as string,
                            styles: { choiceFieldWrapper: styles.radioOptions} 
                            },
                            {
                            key: 'outcomes',
                            text: localization.ModelComparison.disparityInOutcomes,
                            styles: { choiceFieldWrapper: styles.radioOptions}
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
            const disparityMetric = this.props.dashboardContext.modelMetadata.predictionType === PredictionTypes.binaryClassification ?
                ParityOptions[this.props.parityPickerProps.selectedParityKey].parityMetric : "average";
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
            this.setState({featureKey: featureKey, accuracyArray: undefined, disparityArray: undefined});
        }
    }

    private readonly accuracyChanged = (ev: React.FormEvent<HTMLInputElement>, option: IDropdownOption): void => {
        const accuracyKey = option.key.toString();
        if (this.state.accuracyKey !== accuracyKey) {
            this.props.accuracyPickerProps.selectedAccuracyKey = accuracyKey;
            this.setState({accuracyKey: accuracyKey, accuracyArray: undefined});
        }
    }

    private readonly parityChanged = (ev: React.FormEvent<HTMLInputElement>, option: IDropdownOption): void => {
        const parityKey = option.key.toString();
        if (this.state.parityKey !== parityKey) {
            this.props.parityPickerProps.selectedParityKey = parityKey;
            this.setState({parityKey: parityKey, disparityArray: undefined});
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
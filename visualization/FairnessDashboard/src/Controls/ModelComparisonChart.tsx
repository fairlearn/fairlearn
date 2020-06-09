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
import { getTheme, Text } from "office-ui-fabric-react";
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
                                    <svg style={{verticalAlign: "middle", marginRight: "10px"}} width="24" height="28" viewBox="0 0 24 28" fill="none" xmlns="http://www.w3.org/2000/svg">
                                        <path fillRule="evenodd" clipRule="evenodd" d="M11.5812 4.30233C15.9998 4.06977 19.6626 7.61628 19.6626 11.9767C19.6626 13.7791 19.0231 15.4651 17.9184 16.8023C16.5812 18.4884 15.8835 20.3488 15.8835 22.2674V22.6744C15.8835 23.0233 15.651 23.2558 15.3021 23.2558H8.49981C8.15097 23.2558 7.91842 23.0233 7.91842 22.6744V22.3256C7.91842 20.2907 7.27888 18.3721 6.05795 16.9767C4.77888 15.5233 4.13935 13.6047 4.19749 11.6279C4.3719 7.73256 7.62772 4.47674 11.5812 4.30233ZM9.13935 22.093H14.7789C14.7789 20 15.5928 17.907 17.0463 16.1047C17.9766 14.8837 18.4998 13.4884 18.4998 11.9767C18.4998 8.25581 15.4184 5.23256 11.6393 5.40698C8.32539 5.5814 5.59283 8.31395 5.41842 11.686C5.36028 13.314 5.88353 14.9419 6.98818 16.2209C8.32539 17.7907 9.08121 19.8837 9.13935 22.093Z" fill="#666666"/>
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
                                    <Text className={styles.insights} block>{localization.ModelComparison.insights}</Text>   
                                </div>
                                <div className={styles.insightsText}>
                                    <Text className={styles.textSection} block>{insights2}</Text>
                                    <Text className={styles.textSection} block>{insights3}</Text>
                                    <Text className={styles.textSection} block>{insights4}</Text>
                                </div>                                
                                <div className={styles.downloadReport}>
                                <svg style={{verticalAlign: "middle", marginRight: "10px"}} width="17" height="18" viewBox="0 0 17 18" fill="none" xmlns="http://www.w3.org/2000/svg">
                                    <path d="M16.4453 16.972C16.4453 17.3459 16.142 17.6492 15.7682 17.6492L1.47353 17.6492C1.0997 17.6492 0.79641 17.3459 0.79641 16.972C0.79641 16.5982 1.0997 16.2949 1.47353 16.2949H15.7682C15.9478 16.2949 16.1112 16.3649 16.2322 16.4789C16.3633 16.6023 16.4453 16.7775 16.4453 16.972Z" fill="#5A53FF"/>
                                    <path d="M9.57503 11.717L9.57503 0.877332C9.57503 0.499978 9.27728 0.194336 8.90935 0.194336C8.54172 0.194336 8.24397 0.499978 8.24397 0.877332L8.24397 11.717L4.92809 8.34501C4.66561 8.09107 4.25272 8.09811 3.99879 8.36204C3.74455 8.62476 3.74136 9.0474 3.99144 9.31482L8.44205 13.8395C8.70158 14.1034 9.11947 14.1034 9.3787 13.8395L13.8293 9.31482C13.9072 9.23958 13.9654 9.14731 14.0004 9.04566C14.0239 8.97686 14.0368 8.90397 14.038 8.82994C14.0412 8.64534 13.9716 8.46785 13.8455 8.33736C13.7194 8.20627 13.5466 8.13399 13.367 8.13519C13.28 8.13573 13.1951 8.15397 13.1163 8.18749C13.0329 8.22331 12.9565 8.27681 12.8927 8.34501L9.57503 11.717Z" fill="#5A53FF"/>
                                </svg>
                                <span style={{verticalAlign: "middle"}}>{localization.ModelComparison.downloadReport}</span>
                                </div>                                
                            </div>
                        </div>;
        }

        return (
            <Stack className={styles.frame}>
                <div className={styles.header}>
                    <h2 className={styles.headerTitle}>{localization.ModelComparison.title} <b>assessment</b></h2>
                    <Text variant={"large"} className={styles.headerTitle} block>{localization.ModelComparison.title}</Text>
                    {/*<ActionButton iconProps={{iconName: "Edit"}} onClick={this.props.onEditConfigs} className={styles.editButton}>{localization.Report.editConfiguration}</ActionButton>*/}
                </div>
                <div className={styles.headerOptions}>
                    <Dropdown
                        className={styles.dropDown}
                        // label="Feature"
                        defaultSelectedKey={this.props.dashboardContext.modelMetadata.featureNames[this.props.featureBinPickerProps.selectedBinIndex]}
                        options={featureOptions}
                        disabled={false}
                        onChange={this.featureChanged}
                        styles={dropdownStyles}
                    />
                    <Dropdown
                        className={styles.dropDown}
                        // label="Accuracy"
                        defaultSelectedKey={this.props.accuracyPickerProps.selectedAccuracyKey}
                        options={accuracyOptions}
                        disabled={false}
                        onChange={this.accuracyChanged}
                        styles={dropdownStyles}
                    />
                    <Dropdown
                        className={styles.dropDown}
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
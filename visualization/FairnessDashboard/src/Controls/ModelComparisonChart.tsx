import _ from 'lodash';
import { AccessibleChart, ChartBuilder, IPlotlyProperty, PlotlyMode, SelectionContext } from 'mlchartlib';
import {
    getTheme,
    Text,
    Dropdown,
    IDropdownOption,
    IDropdownStyles,
    Modal,
    IIconProps,
    Icon,
} from 'office-ui-fabric-react';
import { ActionButton, PrimaryButton, IconButton } from 'office-ui-fabric-react/lib/Button';
import { ChoiceGroup, IChoiceGroupOption } from 'office-ui-fabric-react/lib/ChoiceGroup';
import { Spinner, SpinnerSize } from 'office-ui-fabric-react/lib/Spinner';
import { Stack } from 'office-ui-fabric-react/lib/Stack';
import React from 'react';
import { AccuracyOptions } from '../AccuracyMetrics';
import { IAccuracyPickerProps, IFeatureBinPickerProps, IParityPickerProps } from '../FairnessWizard';
import { FormatMetrics } from '../FormatMetrics';
import { IFairnessContext } from '../IFairnessContext';
import { PredictionTypes } from '../IFairnessProps';
import { localization } from '../Localization/localization';
import { MetricsCache } from '../MetricsCache';
import { ParityModes, ParityOptions } from '../ParityMetrics';
import { ModelComparisionChartStyles } from './ModelComparisionChart.styles';

const theme = getTheme();
export interface IModelComparisonProps {
    showIntro: boolean;
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
        config: {
            displaylogo: false,
            responsive: true,
            modeBarButtonsToRemove: [
                'toggleSpikelines',
                'hoverClosestCartesian',
                'hoverCompareCartesian',
                'zoom2d',
                'pan2d',
                'select2d',
                'lasso2d',
                'zoomIn2d',
                'zoomOut2d',
                'autoScale2d',
                'resetScale2d',
            ],
        },
        data: [
            {
                datapointLevelAccessors: {
                    customdata: {
                        path: ['index'],
                        plotlyPath: 'customdata',
                    },
                },
                mode: PlotlyMode.textMarkers,
                marker: {
                    size: 14,
                },
                textposition: 'top',
                type: 'scatter',
                xAccessor: 'Accuracy',
                yAccessor: 'Parity',
                hoverinfo: 'text',
            } as any,
        ],
        layout: {
            autosize: true,
            plot_bgcolor: theme.semanticColors.bodyFrameBackground,
            font: {
                size: 10,
            },
            margin: {
                t: 4,
                r: 0,
            },
            hovermode: 'closest',
            xaxis: {
                showgrid: false,
                automargin: true,
                fixedrange: true,
                mirror: true,
                linecolor: theme.semanticColors.disabledBorder,
                linewidth: 1,
                title: {
                    text: 'Error',
                },
            },
            yaxis: {
                showgrid: false,
                automargin: true,
                fixedrange: true,
                title: {
                    text: 'Disparity',
                },
            },
        } as any,
    };

    constructor(props: IModelComparisonProps) {
        super(props);
        this.state = {
            showModalIntro: this.props.showIntro,
            accuracyKey: this.props.accuracyPickerProps.selectedAccuracyKey,
            parityKey: this.props.parityPickerProps.selectedParityKey,
        };
    }

    public render(): React.ReactNode {
        const featureOptions: IDropdownOption[] = this.props.dashboardContext.modelMetadata.featureNames.map((x) => {
            return { key: x, text: x };
        });
        const accuracyOptions: IDropdownOption[] = this.props.accuracyPickerProps.accuracyOptions.map((x) => {
            return { key: x.key, text: x.title };
        });
        const parityOptions: IDropdownOption[] = this.props.parityPickerProps.parityOptions.map((x) => {
            return { key: x.key, text: x.title };
        });

        const dropdownStyles: Partial<IDropdownStyles> = {
            dropdown: { width: 180 },
            title: { borderRadius: '5px' },
        };

        const iconButtonStyles = {
            root: {
                color: theme.semanticColors.bodyText,
                marginLeft: 'auto',
                marginTop: '4px',
                marginRight: '2px',
            },
            rootHovered: {
                color: theme.semanticColors.bodyBackgroundHovered,
            },
        };

        const styles = ModelComparisionChartStyles();

        let mainChart;
        if (!this.state || this.state.accuracyArray === undefined || this.state.disparityArray === undefined) {
            this.loadData();
            mainChart = (
                <Spinner className={styles.spinner} size={SpinnerSize.large} label={localization.calculating} />
            );
        } else {
            const data = this.state.accuracyArray.map((accuracy, index) => {
                return {
                    Parity: this.state.disparityArray[index],
                    Accuracy: accuracy,
                    index: index,
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
            const formattedMinAccuracy = FormatMetrics.formatNumbers(
                minAccuracy,
                this.props.accuracyPickerProps.selectedAccuracyKey,
            );
            const formattedMaxAccuracy = FormatMetrics.formatNumbers(
                maxAccuracy,
                this.props.accuracyPickerProps.selectedAccuracyKey,
            );
            const formattedMinDisparity = FormatMetrics.formatNumbers(
                minDisparity,
                this.props.accuracyPickerProps.selectedAccuracyKey,
            );
            const formattedMaxDisparity = FormatMetrics.formatNumbers(
                maxDisparity,
                this.props.accuracyPickerProps.selectedAccuracyKey,
            );

            let selectedMetric = AccuracyOptions[this.props.accuracyPickerProps.selectedAccuracyKey];

            // handle custom metric case
            if (selectedMetric === undefined) {
                selectedMetric = this.props.accuracyPickerProps.accuracyOptions.find(
                    (metric) => metric.key === this.props.accuracyPickerProps.selectedAccuracyKey,
                );
            }

            const insights2 = localization.formatString(
                localization.ModelComparison.insightsText2,
                selectedMetric.title,
                formattedMinAccuracy,
                formattedMaxAccuracy,
                formattedMinDisparity,
                formattedMaxDisparity,
            );

            const insights3 = localization.formatString(
                localization.ModelComparison.insightsText3,
                selectedMetric.title.toLowerCase(),
                selectedMetric.isMinimization ? formattedMinAccuracy : formattedMaxAccuracy,
                FormatMetrics.formatNumbers(
                    this.state.disparityArray[selectedMetric.isMinimization ? minAccuracyIndex : maxAccuracyIndex],
                    this.props.accuracyPickerProps.selectedAccuracyKey,
                ),
            );

            const insights4 = localization.formatString(
                localization.ModelComparison.insightsText4,
                selectedMetric.title.toLowerCase(),
                FormatMetrics.formatNumbers(
                    this.state.accuracyArray[minDisparityIndex],
                    this.props.accuracyPickerProps.selectedAccuracyKey,
                ),
                formattedMinDisparity,
            );

            const howToReadText = localization.formatString(
                localization.ModelComparison.howToReadText,
                this.props.modelCount.toString(),
                selectedMetric.title.toLowerCase(),
                selectedMetric.isMinimization
                    ? localization.ModelComparison.lower
                    : localization.ModelComparison.higher,
            );

            const props = _.cloneDeep(this.plotlyProps);
            props.data = ChartBuilder.buildPlotlySeries(props.data[0], data).map((series) => {
                series.name = this.props.dashboardContext.modelNames[series.name];
                series.text = this.props.dashboardContext.modelNames;
                return series;
            });

            const accuracyMetricTitle = selectedMetric.title;
            const parityMetricTitle = ParityOptions[this.props.parityPickerProps.selectedParityKey].title;
            props.layout.xaxis.title = accuracyMetricTitle;
            props.layout.yaxis.title = parityMetricTitle;

            const InsightsIcon = () => <Icon iconName="CRMCustomerInsightsApp" className={styles.insightsIcon} />;
            const DownloadIcon = () => <Icon iconName="Download" className={styles.downloadIcon} />;

            const cancelIcon: IIconProps = { iconName: 'Cancel' };

            mainChart = (
                <div className={styles.main}>
                    <div className={styles.mainLeft}>
                        <div className={styles.howTo}>
                            <Modal
                                titleAriaId="intro modal"
                                isOpen={this.state.showModalIntro}
                                onDismiss={this.handleCloseModalIntro}
                                isModeless={true}
                                containerClassName={styles.modalContentIntro}
                            >
                                <div style={{ display: 'flex' }}>
                                    <IconButton
                                        styles={iconButtonStyles}
                                        iconProps={cancelIcon}
                                        ariaLabel="Close intro modal"
                                        onClick={this.handleCloseModalIntro}
                                    />
                                </div>
                                <p className={styles.modalContentIntroText}>
                                    {localization.ModelComparison.introModalText}
                                </p>
                                <div style={{ display: 'flex', paddingBottom: '20px' }}>
                                    <PrimaryButton className={styles.doneButton} onClick={this.handleCloseModalIntro}>
                                        {localization.done}
                                    </PrimaryButton>
                                </div>
                            </Modal>
                            <ActionButton onClick={this.handleOpenModalHelp}>
                                <div className={styles.infoButton}>i</div>
                                {localization.ModelComparison.howToRead}
                            </ActionButton>
                            <Modal
                                titleAriaId="help modal"
                                isOpen={this.state.showModalHelp}
                                onDismiss={this.handleCloseModalHelp}
                                isModeless={true}
                                containerClassName={styles.modalContentHelp}
                            >
                                <div style={{ display: 'flex' }}>
                                    <IconButton
                                        styles={iconButtonStyles}
                                        iconProps={cancelIcon}
                                        ariaLabel="Close popup modal"
                                        onClick={this.handleCloseModalHelp}
                                    />
                                </div>
                                <p className={styles.modalContentHelpText}>
                                    {localization.ModelComparison.helpModalText1}
                                    <br />
                                    <br />
                                    {localization.ModelComparison.helpModalText2}
                                </p>
                                <div style={{ display: 'flex', paddingBottom: '20px' }}>
                                    <PrimaryButton className={styles.doneButton} onClick={this.handleCloseModalHelp}>
                                        {localization.done}
                                    </PrimaryButton>
                                </div>
                            </Modal>
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
                            <Text className={styles.insights} block>
                                {localization.ModelComparison.insights}
                            </Text>
                        </div>
                        <div className={styles.insightsText}>
                            <Text className={styles.textSection} block>
                                {insights2}
                            </Text>
                            <Text className={styles.textSection} block>
                                {insights3}
                            </Text>
                            <Text className={styles.textSection} block>
                                {insights4}
                            </Text>
                        </div>
                        <div className={styles.downloadReport}>
                            <DownloadIcon />
                            <Text style={{ verticalAlign: 'middle' }}>
                                {localization.ModelComparison.downloadReport}
                            </Text>
                        </div>
                    </div>
                </div>
            );
        }

        return (
            <Stack className={styles.frame}>
                <div className={styles.header}>
                    <Text variant={'large'} className={styles.headerTitle} block>
                        {localization.ModelComparison.title} <b>assessment</b>
                    </Text>
                </div>
                <div className={styles.headerOptions}>
                    <Dropdown
                        className={styles.dropDown}
                        defaultSelectedKey={
                            this.props.dashboardContext.modelMetadata.featureNames[
                                this.props.featureBinPickerProps.selectedBinIndex
                            ]
                        }
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
            </Stack>
        );
    }

    private async loadData(): Promise<void> {
        try {
            const accuracyPromises = new Array(this.props.modelCount).fill(0).map((unused, modelIndex) => {
                return this.props.metricsCache.getMetric(
                    this.props.dashboardContext.binVector,
                    this.props.featureBinPickerProps.selectedBinIndex,
                    modelIndex,
                    this.props.accuracyPickerProps.selectedAccuracyKey,
                );
            });
            const parityOption = ParityOptions[this.props.parityPickerProps.selectedParityKey];
            const disparityMetric =
                this.props.dashboardContext.modelMetadata.predictionType === PredictionTypes.binaryClassification
                    ? parityOption.parityMetric
                    : 'average';
            const parityMode = parityOption.parityMode;
            const disparityPromises = new Array(this.props.modelCount).fill(0).map((unused, modelIndex) => {
                return this.props.metricsCache.getDisparityMetric(
                    this.props.dashboardContext.binVector,
                    this.props.featureBinPickerProps.selectedBinIndex,
                    modelIndex,
                    disparityMetric,
                    parityMode,
                );
            });

            const accuracyArray = (await Promise.all(accuracyPromises)).map((metric) => metric.global);
            const disparityArray = await Promise.all(disparityPromises);
            this.setState({ accuracyArray, disparityArray });
        } catch {
            // todo;
        }
    }

    private readonly featureChanged = (ev: React.FormEvent<HTMLInputElement>, option: IDropdownOption): void => {
        const featureKey = option.key.toString();
        if (this.state.featureKey !== featureKey) {
            this.props.featureBinPickerProps.selectedBinIndex = this.props.dashboardContext.modelMetadata.featureNames.indexOf(
                featureKey,
            );
            this.setState({ featureKey: featureKey, accuracyArray: undefined, disparityArray: undefined });
        }
    };

    private readonly accuracyChanged = (ev: React.FormEvent<HTMLInputElement>, option: IDropdownOption): void => {
        const accuracyKey = option.key.toString();
        if (this.state.accuracyKey !== accuracyKey) {
            this.props.accuracyPickerProps.onAccuracyChange(accuracyKey);
            this.setState({ accuracyKey: accuracyKey, accuracyArray: undefined });
        }
    };

    private readonly parityChanged = (ev: React.FormEvent<HTMLInputElement>, option: IDropdownOption): void => {
        const parityKey = option.key.toString();
        if (this.state.parityKey !== parityKey) {
            this.props.parityPickerProps.onParityChange(parityKey);
            this.setState({ parityKey: parityKey, disparityArray: undefined });
        }
    };

    private readonly handleCloseModalIntro = (event): void => {
        this.setState({ showModalIntro: false });
        this.props.onHideIntro();
    };

    private readonly handleOpenModalHelp = (event): void => {
        this.setState({ showModalHelp: true });
    };

    private readonly handleCloseModalHelp = (event): void => {
        this.setState({ showModalHelp: false });
    };
}

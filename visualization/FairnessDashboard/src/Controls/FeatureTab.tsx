import { INumericRange, RangeTypes } from "mlchartlib";
import { ActionButton } from "office-ui-fabric-react/lib/Button";
import { Icon } from "office-ui-fabric-react/lib/Icon";
import { List } from "office-ui-fabric-react/lib/List";
import { Modal } from 'office-ui-fabric-react/lib/Modal';
import { Stack, StackItem } from "office-ui-fabric-react/lib/Stack";
import { Text, themeRulesStandardCreator, IProcessedStyleSet } from "office-ui-fabric-react";
import React from "react";
import { IBinnedResponse } from "../IBinnedResponse";
import { IWizardTabProps } from "../IWizardTabProps";
import { localization } from "../Localization/localization";
import BinDialog from "./BinDialog";
import { DataSpecificationBlade } from "./DataSpecificationBlade";
import { WizardFooter } from "./WizardFooter";
import { FeatureTabStyles, IFeatureTabStyles } from "./FeatureTab.styles";

interface IFeatureItem {
    title: string;
    description: string;
    onSelect: (index: number) => void;
    selected: boolean;
    categories?: string[];
}

export interface IFeatureTabProps extends IWizardTabProps {
    featureBins: IBinnedResponse[];
    selectedFeatureIndex: number;
    selectedFeatureChange: (value: number) => void;
    saveBin: (bin: IBinnedResponse) => void;
}

interface IState {
    expandedBins: number[];
    editingFeatureIndex: number | undefined;
}

export class FeatureTab extends React.PureComponent<IFeatureTabProps, IState> {
    // private static readonly classNames = mergeStyleSets({
    //     itemCell: {
    //         display: "flex",
    //         flexDirection: "row",
    //         padding: "20px 0",
    //         width: "100%",
    //         cursor: "pointer",
    //         boxSizing: "border-box",
    //         borderBottom: "1px solid #CCCCCC",
    //         selectors: {
    //           '&:hover': { background: "lightgray" }
    //         }
    //     },
    //     iconClass: {
    //         fontSize: "20px"
    //     },
    //     itemsList: {
    //         overflowY: "auto"
    //     },
    //     frame: {
    //         height: "100%",
    //     },
    //     main: {
    //         height: "100%",
    //         maxWidth: "700px",
    //         flex: 1
    //     },
    //     header: {
    //         color: "#333333",
    //         fontSize: "32px",
    //         lineHeight: "39px",
    //         fontWeight: "100"
    //     },
    //     textBody: {
    //         paddingTop: "12px",
    //         fontSize: "18px",
    //         lineHeight: "24px",
    //         fontWeight: "300",
    //         marginBottom: "15px"
    //     },
    //     tableHeader: {
    //         display: "flex",
    //         flexDirection: "row",
    //         justifyContent: "space-between",
    //         paddingBottom: "15px",
    //         color: "#333333",
    //         fontSize: "15px",
    //         lineHeight: "18px",
    //         fontWeight: "500",
    //         borderBottom: "1px solid #CCCCCC"
    //     },
    //     itemTitle: {
    //         margin: 0,
    //         color: "#333333",
    //         fontSize: "22px",
    //         lineHeight: "26px",
    //         fontWeight: "300"
    //     },
    //     valueCount: {
    //         paddingTop: "15px",
    //         color: "#333333",
    //         fontSize: "15px",
    //         lineHeight: "18px",
    //         fontWeight: "500"
    //     },
    //     iconWrapper: {
    //         paddingTop: "4px",
    //         paddingLeft: "5px",
    //         width: "30px"
    //     },
    //     featureDescriptionSection: {
    //         flex: 1,
    //         paddingRight: "20px",
    //         minHeight:"75px"
    //     },
    //     binSection:{
    //         width:"130px",

    //     },
    //     expandButton: {
    //         paddingLeft: 0,
    //         selectors: {
    //             "& i":{
    //                 marginLeft: 0
    //             }
    //         }
    //     },
    //     category: {
    //         textOverflow: "ellipsis",
    //         whiteSpace: "nowrap",
    //         overflow: "hidden"
    //     },
    //     subgroupHeader: {
    //         width: "130px"
    //     }
    // });

    constructor(props: IFeatureTabProps) {
        super(props);
        this.state = { 
            expandedBins: [],
            editingFeatureIndex: undefined
        };
    }
    
    render(): React.ReactNode {
        const styles = FeatureTabStyles();
        return(
            <Stack horizontal horizontalAlign="space-between" className={styles.frame}>
                <Modal
                    isOpen={this.state.editingFeatureIndex !== undefined}
                    isBlocking={false}
                    onDismiss={this.hideModal}>
                    { this.state.editingFeatureIndex !== undefined && <BinDialog
                        range={this.props.dashboardContext.modelMetadata.featureRanges[this.state.editingFeatureIndex] as INumericRange}
                        bins={this.props.featureBins[this.state.editingFeatureIndex]}
                        dataset={this.props.dashboardContext.dataset}
                        index={this.state.editingFeatureIndex}
                        onSave={this.onBinSave}
                        onCancel={this.hideModal}/>}
                </Modal>
                <Stack className={styles.main}>
                    <h2 className={styles.header}>
                        {localization.Feature.header}
                    </h2>
                    <p className={styles.textBody}>{localization.Feature.body}</p>
                    <div className={styles.tableHeader}>
                        {/* <div>{localization.Intro.features}</div> */}
                        <Text variant={"mediumPlus"} block>{localization.Intro.features}</Text>
                        {/* <div className={styles.subgroupHeader}>{localization.Feature.subgroups}</div> */}
                        <Text className={styles.subgroupHeader} block>{localization.Feature.subgroups}</Text>
                    </div>
                    <StackItem grow={2} className={styles.itemsList}>
                        <List
                            items={this.props.featureBins}
                            onRenderCell={this._onRenderCell.bind(this, styles)}
                        />
                    </StackItem>
                    <WizardFooter onNext={this.props.onNext}/>
                </Stack>
                <DataSpecificationBlade
                        numberRows={this.props.dashboardContext.trueY.length}
                        featureNames={this.props.dashboardContext.modelMetadata.featureNames}/>
            </Stack>
        );
    }

    private readonly hideModal = (): void => {
        this.setState({editingFeatureIndex: undefined});
    }

    private readonly onBinSave = (bin: IBinnedResponse): void => {
        this.setState({editingFeatureIndex: undefined});
        this.props.saveBin(bin);
    }

    private readonly editBins = (index: number) => {
        this.setState({editingFeatureIndex: index});
    }

    private readonly _onRenderCell = (styles: IProcessedStyleSet<IFeatureTabStyles>, item: IBinnedResponse, index: number | undefined): JSX.Element => {
        //debugger;
        return (
          <div
            key={index}
            className={styles.itemCell}
            onClick={this.props.selectedFeatureChange.bind(this, index)}
            data-is-focusable={true}
          >
            <div className={styles.iconWrapper}>
                <Icon iconName={this.props.selectedFeatureIndex === index ? "RadioBtnOn" : "RadioBtnOff"} className={styles.iconClass}/>
            </div>
            {/* <Text className={styles.iconWrapper} block>
                <Icon iconName={this.props.selectedFeatureIndex === index ? "RadioBtnOn" : "RadioBtnOff"} className={styles.iconClass}/>
            </Text> */}
            <div className={styles.featureDescriptionSection}>
                {/* <h2 className={styles.itemTitle}>{this.props.dashboardContext.modelMetadata.featureNames[index]}</h2> */}
                <Text variant={"large"} className={styles.itemTitle} block>{this.props.dashboardContext.modelMetadata.featureNames[index]}</Text>
                {item.rangeType === RangeTypes.categorical &&
                    // <div className={styles.valueCount}>{localization.formatString(localization.Feature.summaryCategoricalCount, item.array.length) as string}</div>
                    <Text variant={"mediumPlus"} className={styles.valueCount} block>{localization.formatString(localization.Feature.summaryCategoricalCount, item.array.length) as string}</Text>
                }
                {item.rangeType !== RangeTypes.categorical &&
                    // <div className={styles.valueCount}>
                    // {localization.formatString(localization.Feature.summaryNumericCount, 
                    //     (this.props.dashboardContext.modelMetadata.featureRanges[index] as INumericRange).min, 
                    //     (this.props.dashboardContext.modelMetadata.featureRanges[index] as INumericRange).max, 
                    //     item.labelArray.length) as string}</div>

                    <Text variant={"mediumPlus"} className={styles.valueCount} block>
                    {localization.formatString(localization.Feature.summaryNumericCount, 
                        (this.props.dashboardContext.modelMetadata.featureRanges[index] as INumericRange).min, 
                        (this.props.dashboardContext.modelMetadata.featureRanges[index] as INumericRange).max, 
                        item.labelArray.length) as string}</Text>
                }
                {!this.props.dashboardContext.modelMetadata.featureIsCategorical[index] && 
                    <ActionButton 
                        className={styles.expandButton}
                        iconProps={{iconName: "Edit"}}
                        onClick={this.editBins.bind(this, index)}>{localization.Feature.editBinning}</ActionButton>
                }
            </div>
            <div className={styles.binSection}>
                {!this.state.expandedBins.includes(index) && !!item.labelArray && 
                    <div> 
                        {item.labelArray.slice(0,7).map((category, index) => <div key={index} className={styles.category}>{category}</div>)}
                        {item.labelArray.length > 7 && <ActionButton
                            className={styles.expandButton}
                            iconProps={{iconName: "ChevronDownMed"}}
                            onClick={this.updateExpandedList.bind(this, index)}>{localization.Feature.showCategories}</ActionButton>}
                    </div>
                }
                {this.state.expandedBins.includes(index) && !!item.labelArray &&
                <div>
                    {item.labelArray.map((category, index) => <div key={index} className={styles.category}>{category}</div>)}
                    {<ActionButton 
                        className={styles.expandButton}
                        iconProps={{iconName: "ChevronUpMed"}}
                        onClick={this.updateExpandedList.bind(this)}>{localization.Feature.hideCategories}</ActionButton>}
                </div>
                }
            </div>
        </div>
        );
    }

    private readonly updateExpandedList = (value?: number): void => {
        this.setState(() => {return {expandedBins: [value]}})
    }
}
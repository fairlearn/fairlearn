import _ from "lodash";
import { INumericRange, RangeTypes } from "mlchartlib";
import { DefaultButton, PrimaryButton } from "office-ui-fabric-react/lib/Button";
import { Checkbox } from 'office-ui-fabric-react/lib/Checkbox';
import { SpinButton } from 'office-ui-fabric-react/lib/SpinButton';
import { Text } from 'office-ui-fabric-react';
import React from "react";
import { BinnedResponseBuilder } from "../BinnedResponseBuilder";
import { IBinnedResponse } from "../IBinnedResponse";
import { localization } from "../Localization/localization";
import { BinDialogStyles } from "./BinDialog.styles";

export interface IBinDialogProps {
    range: INumericRange;
    bins: IBinnedResponse;
    dataset: any[][];
    index: number;
    onSave: (bins: IBinnedResponse) => void;
    onCancel: () => void;
}

export default class BinDialog extends React.PureComponent<IBinDialogProps, IBinnedResponse> {
    // private static readonly classNames = mergeStyleSets({
    //     frame: {
    //         height: "400px",
    //         width: "500px",
    //         display: "flex",
    //         flexDirection: "column",
    //         backgroundColor: "#F2F2F2"
    //     },
    //     header: {
    //         padding: "12px",
    //         textAlign: "center",
    //         backgroundColor: "#333333",
    //         color: "#FFFFFF",
    //         fontSize: "24px",
    //         lineHeight: "30px",
    //         fontWeight: "100"
    //     },
    //     buttons: {
    //         display: "inline-flex",
    //         flexDirection: "row-reverse",
    //         padding: "10px"
    //     },
    //     saveButton: {
    //         height: "44px",
    //         padding: "12px",
    //         color: "#FFFFFF",
    //         fontSize: "18px",
    //         lineHeight: "24px",
    //         backgroundColor: "#666666",
    //         fontWeight: "400",
    //         marginLeft: "10px"
    //     },
    //     cancelButton: {
    //         height: "44px",
    //         padding: "12px",
    //         color: "#333333",
    //         fontSize: "18px",
    //         lineHeight: "24px",
    //         backgroundColor: "#FFFFFF",
    //         fontWeight: "400"
    //     },
    //     binCounter: {
    //         selectors: {
    //             "& label": {
    //                 color: "#333333",
    //                 fontSize: "15px",
    //                 lineHeight: "20px",
    //                 fontWeight: "500"
    //             }
    //         }
    //     },
    //     main: {
    //         flexGrow: 1,
    //         padding: "20px 40px",
    //         overflow: "hidden",
    //         display: "flex",
    //         flexDirection: "column"
    //     },
    //     categoryHeader: {

    //     },
    //     checkbox: {
    //         selectors: {
    //             "& span": {
    //                 color: "#333333",
    //                 fontSize: "15px",
    //                 lineHeight: "20px",
    //                 fontWeight: "500"
    //             }
    //         }
    //     },
    //     controls: {
    //         display: "inline-flex",
    //         width: "100%",
    //         justifyContent: "space-between",
    //         height: "30px",
    //         alignItems: "center",
    //         paddingBottom: "10px",
    //         borderBottom: "1px solid #CCCCCC",
    //         marginBottom: "10px"
    //     },
    //     scrollArea: {
    //         overflowY: "auto",
    //         overflowX: "hidden",
    //         flexGrow:"2"
    //     },
    //     groupLabel: {
    //         color: "#333333",
    //         fontSize: "15px",
    //         lineHeight: "20px",
    //         fontWeight: "400",
    //         height: "25px",
    //         borderBottom: "1px solid #CCCCCC",
    //         paddingLeft: "12px"
    //     }
    // });

    private static minBins = 1;
    private static maxBins = 30;

    constructor(props: IBinDialogProps) {
        super(props);
        this.state = _.cloneDeep(props.bins);
    }

    public render(): React.ReactNode {
        const styles = BinDialogStyles();
        return (
            <div className={styles.frame}>
                {/* <div className={styles.header}>{localization.BinDialog.header}</div> */}
                <Text variant={"xLargePlus"} className={styles.header}>{localization.BinDialog.header}</Text>
                <div className={styles.main}>
                    <div className={styles.controls}>
                        {this.props.range.rangeType === RangeTypes.integer &&
                            <Checkbox
                                //className={styles.checkbox}
                                label={localization.BinDialog.makeCategorical}
                                checked={this.state.rangeType === RangeTypes.categorical}
                                onChange={this.toggleCategorical}/>
                        }
                        {this.state.rangeType !== RangeTypes.categorical &&
                            <div className={styles.binCounter}>
                                <SpinButton
                                    styles={{
                                        spinButtonWrapper: {maxWidth: "98px"},
                                        labelWrapper: { alignSelf: "center"},
                                        root: {
                                            display: "inline-flex",
                                            float: "right",
                                            selectors: {
                                                "> div": {
                                                    maxWidth: "108px"
                                                }
                                            }
                                        }
                                    }}
                                    label={localization.BinDialog.numberOfBins}
                                    min={BinDialog.minBins}
                                    max={BinDialog.maxBins}
                                    value={this.state.array.length.toString()}
                                    onIncrement={this.setBinCount.bind(this, 1)}
                                    onDecrement={this.setBinCount.bind(this, -1)}
                                    onValidate={this.setBinCount.bind(this, 0)}
                                />
                            </div>
                        }
                    </div>
                    <Text>{localization.BinDialog.categoryHeader}</Text>
                    <div className={styles.scrollArea}>
                        {this.state.labelArray.map((val, i) => {
                            return <div className={styles.groupLabel} key={i}>{val}</div>;
                        })}
                    </div>
                </div>
                <div className={styles.buttons}>
                    <PrimaryButton className={styles.saveButton} text={localization.BinDialog.save} onClick={this.onSave}/>
                    <DefaultButton
                        className={styles.cancelButton}
                        text={localization.BinDialog.cancel}
                        onClick={this.props.onCancel}/>
                </div>
            </div>
        );
    }

    private readonly onSave = (): void => {
        this.props.onSave(this.state);
    }

    private readonly toggleCategorical = (ev: React.FormEvent<HTMLElement>, checked: boolean): void => {
        if (checked) {
            this.setState(BinnedResponseBuilder.buildCategorical(this.props.range, this.props.index, this.props.dataset));
        } else {
            if (this.props.bins.rangeType === RangeTypes.integer) {
                this.setState(this.props.bins);
            } else {
                this.setState(BinnedResponseBuilder.buildNumeric(this.props.range, this.props.index, this.props.dataset));
            }
        }
    }

    private readonly setBinCount = (delta: number, stringVal: string): string | void => {
        if (delta === 0) {
            const number = +stringVal;
            if (!Number.isInteger(number) || number > BinDialog.maxBins || number < BinDialog.minBins) {
                return this.state.array.length.toString();
            }
            this.setState(BinnedResponseBuilder.buildNumeric(this.props.range, this.props.index, this.props.dataset, number));
        } else {
            const prevVal = this.state.array.length;
            const binCount = prevVal + delta;
            if (binCount > BinDialog.maxBins || binCount < BinDialog.minBins) {
                return prevVal.toString();
            }
            this.setState(BinnedResponseBuilder.buildNumeric(this.props.range, this.props.index, this.props.dataset, prevVal + delta));
        }
    }
}
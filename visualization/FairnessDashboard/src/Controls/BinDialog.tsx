import React from "react";
import { INumericRange, RangeTypes } from "mlchartlib";
import _ from "lodash";
import { mergeStyleSets } from "@uifabric/styling";
import { Checkbox, ICheckboxProps } from 'office-ui-fabric-react/lib/Checkbox';
import { SpinButton } from 'office-ui-fabric-react/lib/SpinButton';
import { localization } from "../Localization/localization";
import { PrimaryButton, DefaultButton } from "office-ui-fabric-react/lib/Button";
import { IBinnedResponse } from "../IBinnedResponse";
import { BinnedResponseBuilder } from "../BinnedResponseBuilder";

export interface IBinDialogProps {
    range: INumericRange;
    bins: IBinnedResponse;
    dataset: any[][];
    index: number;
    onSave: (bins: IBinnedResponse) => void;
    onCancel: () => void;
}

export default class BinDialog extends React.PureComponent<IBinDialogProps, IBinnedResponse> {
    private static readonly classNames = mergeStyleSets({
        header: {
            color: "#333333",
            fontSize: "24px",
            lineHeight: "30px",
            fontWeight: "100"
        },
        buttons: {
            display: "inline-flex",
            flexDirection: "row-reverse",
            paddingTop: "10px",
            paddingBottom: "10px"
        },
        saveButton: {
            height: "44px",
            padding: "12px",
            color: "#FFFFFF",
            fontSize: "18px",
            lineHeight: "24px",
            backgroundColor: "#666666",
            fontWeight: "400",
            marginLeft: "10px"
        },
        cancelButton: {
            height: "44px",
            padding: "12px",
            color: "#333333",
            fontSize: "18px",
            lineHeight: "24px",
            backgroundColor: "#FFFFFF",
            fontWeight: "400"
        },
        binCounter: { }
    });

    private static minBins = 1;
    private static maxBins = 30;

    constructor(props: IBinDialogProps) {
        super(props);
        this.state = _.cloneDeep(props.bins);
    }

    public render(): React.ReactNode {
        return (
            <div>
                <div className={BinDialog.classNames.header}>{localization.BinDialog.header}</div>
                {this.props.range.rangeType === RangeTypes.integer &&
                    <Checkbox
                        label={localization.BinDialog.makeCategorical}
                        checked={this.state.rangeType === RangeTypes.categorical}
                        onChange={this.toggleCategorical}/>
                }
                {this.state.rangeType !== RangeTypes.categorical &&
                    <div>
                        <div className={BinDialog.classNames.binCounter}>
                            <div>{localization.BinDialog.numberOfBins}</div>
                            <SpinButton
                                min={BinDialog.minBins}
                                max={BinDialog.maxBins}
                                value={this.state.array.length.toString()}
                                onIncrement={this.setBinCount.bind(this, 1)}
                                onDecrement={this.setBinCount.bind(this, -1)}
                                onValidate={this.setBinCount.bind(this, 0)}
                            />
                        </div>
                        {this.state.labelArray.map(val => {
                            return <div>{val}</div>;
                        })}
                    </div>
                }
                <div className={BinDialog.classNames.buttons}>
                    <PrimaryButton className={BinDialog.classNames.saveButton} text={localization.BinDialog.save} onClick={this.onSave}/>
                    <DefaultButton
                        className={BinDialog.classNames.cancelButton}
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
import { IAccuracyPickerProps } from "../FairnessWizard";
import React from "react";
import { localization } from "../Localization/localization";
import { IconButton, DefaultButton } from "office-ui-fabric-react/lib/Button";
import { ComboBox, IComboBox, IComboBoxOption } from "office-ui-fabric-react/lib/ComboBox";
import { IDropdownOption } from "office-ui-fabric-react/lib/Dropdown";
import { Callout } from "office-ui-fabric-react/lib/Callout";

interface IState {
    showCallout: boolean;
}

export class AccuracyPicker extends React.PureComponent <IAccuracyPickerProps, IState> {
    private _accuracyDropdownHelpId: string = "_accuracyDropdownHelpId";
    constructor(props: IAccuracyPickerProps) {
        super(props);
        this.state = { showCallout: false};
    }
    render(): React.ReactNode {
        const options: IDropdownOption[] = this.props.accuracyOptions.map(option => {
            return {
                key: option.key,
                text: option.title
            };
        });
        return (<div>
            <div className="selector">
                <div className="selector-label">
                    <span>{"TODO"}</span>
                    <IconButton
                        id={this._accuracyDropdownHelpId}
                        iconProps={{ iconName: 'Info' }}
                        title={"TODO"}
                        ariaLabel="Info"
                        onClick={this.onOpen}
                        styles={{ root: { marginBottom: -3, color: 'rgb(0, 120, 212)' } }}
                    />
                </div>
                <ComboBox
                    className="path-selector"
                    selectedKey={this.props.selectedAccuracyKey}
                    onChange={this.onAccuracyChange}
                    options={options}
                    ariaLabel={"Accuracy selector"}
                    useComboBoxAsMenuWidth={true}
                /> 
            </div>
            {this.state.showCallout && (
                <Callout
                    target={'#' + this._accuracyDropdownHelpId}
                    setInitialFocus={true}
                    onDismiss={this.onDismiss}
                    role="alertdialog">
                    <div className="callout-info">
                        <DefaultButton onClick={this.onDismiss}>{localization.close}</DefaultButton>
                    </div>
                </Callout>
                )}
        </div>);
    }

    private readonly onDismiss = (): void => {
        this.setState({ showCallout: false});
    }

    private readonly onOpen = (): void => {
        this.setState({ showCallout: true});
    }

    private readonly onAccuracyChange = (event: React.FormEvent<IComboBox>, item: IComboBoxOption) => {
        this.props.onAccuracyChange(item.key as string);
    }
}
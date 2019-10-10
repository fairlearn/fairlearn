import React from "react";
import { DefaultButton, PrimaryButton } from "office-ui-fabric-react/lib/Button";
import { localization } from "../Localization/localization";

export interface IWizardFooterProps {
    onNext: () => void;
    onPrevious?: () => void;
}

export class WizardFooter extends React.PureComponent<IWizardFooterProps>{
    public render(): React.ReactNode {
        return (
            <div>
                {(!!this.props.onPrevious) &&
                    <DefaultButton text={localization.Footer.back} onClick={this.props.onPrevious}/>
                }
                <PrimaryButton text={localization.Footer.next} onClick={this.props.onNext}/>
            </div>
        );
    }
}
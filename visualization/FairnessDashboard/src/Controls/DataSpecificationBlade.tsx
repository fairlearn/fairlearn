import React from "react";
import { Text } from "office-ui-fabric-react/lib/Text";
import { localization } from "../Localization/localization";
import { Separator } from "office-ui-fabric-react/lib/Separator";

export interface IDataSpecProps {
    numberRows: number;
    featureNames: string[];
}

export class DataSpecificationBlade extends React.PureComponent<IDataSpecProps> {
    render(): React.ReactNode {
        return (<div style={{width: "250px", margin: "25px"}}>
            <div style={{fontWeight: "bold"}}>
                {localization.dataSpecifications}
            </div>
            <Text block={true}>
                {localization.formatString(localization.attributesCount, this.props.featureNames.length)}
            </Text>
            <Text block={true}>
                {localization.formatString(localization.instanceCount, this.props.numberRows)}
            </Text>
            <Separator />
            <div style={{fontWeight: "bold"}}>
                {localization.attributes}
            </div>
            {this.props.featureNames.map(name => {return <Text block={true}>{name}</Text>;})}
        </div>);
    }
}
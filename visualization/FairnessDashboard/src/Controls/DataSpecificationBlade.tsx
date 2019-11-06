import React from "react";
import { Text } from "office-ui-fabric-react/lib/Text";
import { localization } from "../Localization/localization";
import { Separator } from "office-ui-fabric-react/lib/Separator";
import { mergeStyleSets } from "@uifabric/styling";

export interface IDataSpecProps {
    numberRows: number;
    featureNames: string[];
}

export class DataSpecificationBlade extends React.PureComponent<IDataSpecProps> {
    private static readonly classNames = mergeStyleSets({
        title: {
            color: "#333333",
            fontSize: "12px",
            lineHeight: "15px",
            fontWeight: "500",
            height: "20px",
            paddingBottom: "10px"
        },
        frame: {
            paddingLeft:"60px",
            width: "120px",
            boxSizing: "content-box"
        },
        text: {
            fontSize: "12px",
            lineHeight: "15px",
        }
    });
    render(): React.ReactNode {
        return (
        <div className={DataSpecificationBlade.classNames.frame}>
            <div className={DataSpecificationBlade.classNames.title}>
                {localization.dataSpecifications}
            </div>
            <div className={DataSpecificationBlade.classNames.text}>
                {this.props.featureNames.length === 1 ?
                    localization.singleAttributeCount :
                    localization.formatString(localization.attributesCount, this.props.featureNames.length)}
            </div>
            <div className={DataSpecificationBlade.classNames.text}>
                {localization.formatString(localization.instanceCount, this.props.numberRows)}
            </div>
        </div>);
    }
}
import { getTheme, IProcessedStyleSet, IStyle, mergeStyleSets, FontWeights, FontSizes } from 'office-ui-fabric-react';

export interface IOverallTableStyles {
    minMaxLabel: IStyle;
    groupCol: IStyle;
    groupLabel: IStyle;
    flexCol: IStyle;
    binBox: IStyle;
    binTitle: IStyle;
    binLabel: IStyle;
    sensitiveAttributes: IStyle;
    metricCol: IStyle;
    metricLabel: IStyle;
    metricBox: IStyle;
    frame: IStyle;
}

export const OverallTableStyles: () => IProcessedStyleSet<IOverallTableStyles> = () => {
    const theme = getTheme();
    return mergeStyleSets<IOverallTableStyles>({
        minMaxLabel: {
            padding: '1px 9px',
            marginTop: '4px',
            fontSize: '10px',
            lineHeight: '20px',
            fontWeight: '400',
            backgroundColor: theme.semanticColors.bodyBackground,
        },
        groupCol: {
            display: 'inline-flex',
            flexDirection: 'column',
            height: '100%',
            width: 'max-content',
        },
        groupLabel: {
            fontSize: '12px',
            lineHeight: '12px',
            fontWeight: '500',
            height: '26px',
        },
        flexCol: {
            display: 'flex',
            flex: 1,
            flexDirection: 'column',
            borderTop: '0.5px solid',
            borderTopColor: theme.semanticColors.inputBorder,
        },
        binBox: {
            flex: 1,
            width: '130px',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            borderBottom: '0.5px solid',
            borderBottomColor: theme.semanticColors.inputBorder,
        },
        binTitle: {
            fontSize: '12px',
            lineHeight: '16px',
            fontWeight: '600',
        },
        binLabel: {
            fontSize: '12px',
            lineHeight: '16px',
            fontWeight: 'normal',
        },
        sensitiveAttributes: {
            fontSize: '12px',
            lineHeight: '16px',
            fontWeight: 'normal',
            height: '26px',
            paddingLeft: '10px',
        },
        metricCol: {
            display: 'inline-flex',
            flexDirection: 'column',
            height: '100%',
            width: '120px',
        },
        metricLabel: {
            fontSize: '11px',
            lineHeight: '16px',
            fontWeight: '600',
            height: '26px',
            paddingLeft: '10px',
        },
        metricBox: {
            flex: 1,
            paddingLeft: '10px',
            fontSize: '12px',
            lineHeight: '16px',
            fontWeight: 'normal',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            borderBottom: '0.5px solid',
            borderBottomColor: theme.semanticColors.inputBorder,
        },
        frame: {
            paddingBottom: '19px',
            display: 'flex',
        },
    });
};
